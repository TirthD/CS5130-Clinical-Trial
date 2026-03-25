"""
Eligibility Parser Tool
=======================
Extracts structured eligibility criteria from unstructured clinical trial
text using Google Gemini, then checks if a user profile qualifies.

Pipeline:
    1. Raw eligibility text → Gemini → Structured JSON criteria
    2. Structured criteria + User profile → Eligibility check result

The LLM handles the messy NLP work (parsing varied text formats), while
deterministic code handles the matching logic (age comparisons, gender checks).

Usage:
    parser = EligibilityParser()

    # Step 1: Parse raw text into structured criteria
    criteria = parser.parse_eligibility(raw_text)

    # Step 2: Check if a user qualifies
    result = parser.check_eligibility(criteria, user_profile)
    print(result.is_eligible)   # True/False
    print(result.reasons)       # ["Age 45 is within range 18-65", ...]
"""

import json
import os
import re
import logging
from dataclasses import dataclass, field
from enum import Enum

# Lazy import — google.generativeai is only needed when calling Gemini
genai = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Gender(str, Enum):
    ALL = "all"
    MALE = "male"
    FEMALE = "female"


@dataclass
class EligibilityCriteria:
    """Structured representation of a trial's eligibility requirements."""
    min_age: int | None = None              # Minimum age in years
    max_age: int | None = None              # Maximum age in years
    gender: Gender = Gender.ALL             # Required gender
    inclusion_criteria: list[str] = field(default_factory=list)   # Must-have conditions
    exclusion_criteria: list[str] = field(default_factory=list)   # Disqualifying conditions
    required_medications: list[str] = field(default_factory=list) # Current meds required
    excluded_medications: list[str] = field(default_factory=list) # Meds that disqualify
    other_requirements: list[str] = field(default_factory=list)   # Anything else extracted
    raw_text: str = ""                      # Original text for reference
    parse_confidence: float = 0.0           # How confident the parse is (0-1)


@dataclass
class EligibilityResult:
    """Result of checking a user profile against eligibility criteria."""
    is_eligible: bool = False
    met_criteria: list[str] = field(default_factory=list)       # Criteria the user satisfies
    unmet_criteria: list[str] = field(default_factory=list)     # Criteria the user fails
    uncertain_criteria: list[str] = field(default_factory=list) # Can't determine from profile
    summary: str = ""                                           # Human-readable summary


@dataclass
class UserProfile:
    """User's self-reported information for eligibility matching."""
    age: int | None = None
    gender: str | None = None           # "male", "female", or None
    conditions: list[str] = field(default_factory=list)   # e.g. ["type 2 diabetes", "hypertension"]
    medications: list[str] = field(default_factory=list)   # e.g. ["metformin", "lisinopril"]


# ---------------------------------------------------------------------------
# Gemini prompt for structured extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a clinical trial eligibility parser. Extract structured eligibility criteria from the following clinical trial eligibility text.

Return ONLY valid JSON with this exact structure (no markdown, no backticks, no explanation):
{{
    "min_age": <integer or null>,
    "max_age": <integer or null>,
    "gender": "<all | male | female>",
    "inclusion_criteria": ["criterion 1", "criterion 2"],
    "exclusion_criteria": ["criterion 1", "criterion 2"],
    "required_medications": ["medication 1"],
    "excluded_medications": ["medication 1"],
    "other_requirements": ["requirement 1"],
    "parse_confidence": <float 0.0-1.0>
}}

Rules:
- Ages must be integers in years. Convert "18 years" to 18, "6 months" to 0.
- If no age limit is mentioned, set to null.
- Gender should be "all" unless explicitly restricted.
- Keep each criterion as a short, clear phrase.
- Inclusion = what the patient MUST have to join.
- Exclusion = what DISQUALIFIES a patient.
- Required medications = drugs the patient must currently be on.
- Excluded medications = drugs the patient must NOT be taking.
- parse_confidence: 1.0 if text is clear and structured, lower if ambiguous.
- Return ONLY the JSON object. No other text.

Eligibility text to parse:
\"\"\"
{eligibility_text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class EligibilityParser:
    """
    Parses unstructured eligibility text from ClinicalTrials.gov into
    structured criteria, then checks user profiles against them.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        """
        Initialize the parser with Gemini credentials.

        Args:
            api_key: Google Gemini API key. Falls back to config.py / env var.
            model_name: Gemini model to use. Defaults to config setting.
        """
        # Explicitly load .env from project root (2 levels up from this file)
        # so it works regardless of which directory the script is run from
        try:
            from dotenv import load_dotenv
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            dotenv_path = os.path.join(project_root, ".env")
            load_dotenv(dotenv_path, override=True)
            logger.debug(f"Loaded .env from: {dotenv_path}")
        except ImportError:
            logger.debug("python-dotenv not installed, relying on system env vars")

        # Now import config (env vars are loaded, so config picks them up)
        try:
            import config
            self._api_key = api_key or config.GEMINI_API_KEY
            self._model_name = model_name or config.GEMINI_MODEL
            self._max_tokens = getattr(config, "GEMINI_MAX_TOKENS", 2048)
            self._temperature = getattr(config, "GEMINI_TEMPERATURE", 0.1)
        except ImportError:
            self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            self._model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self._max_tokens = 2048
            self._temperature = 0.1

        if not self._api_key:
            logger.warning(
                "No Gemini API key provided. Set GEMINI_API_KEY in .env "
                "or pass api_key to EligibilityParser(). "
                "Regex fallback will be used for parsing."
            )
            self._model = None
            return

        # Import and configure Gemini only when we have a key
        try:
            import google.generativeai as _genai
            _genai.configure(api_key=self._api_key)
            self._model = _genai.GenerativeModel(
                self._model_name,
                generation_config={
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_tokens,
                },
            )
        except ImportError:
            logger.warning(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
            self._model = None

    # ------------------------------------------------------------------
    # Main parsing method
    # ------------------------------------------------------------------

    def parse_eligibility(self, raw_text: str) -> EligibilityCriteria:
        """
        Parse raw eligibility text into structured criteria using Gemini.

        Falls back to regex-based extraction if the LLM call fails.

        Args:
            raw_text: Unstructured eligibility text from ClinicalTrials.gov

        Returns:
            EligibilityCriteria with extracted fields
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty eligibility text provided")
            return EligibilityCriteria(raw_text=raw_text, parse_confidence=0.0)

        # Try LLM-based extraction first
        try:
            criteria = self._parse_with_gemini(raw_text)
            logger.info(f"Gemini parse successful (confidence: {criteria.parse_confidence:.2f})")
            return criteria
        except Exception as e:
            logger.warning(f"Gemini parsing failed: {e}. Falling back to regex.")

        # Fallback: regex-based extraction for basic fields
        try:
            criteria = self._parse_with_regex(raw_text)
            logger.info("Regex fallback parse completed")
            return criteria
        except Exception as e:
            logger.error(f"Regex fallback also failed: {e}")
            return EligibilityCriteria(raw_text=raw_text, parse_confidence=0.0)

    # ------------------------------------------------------------------
    # Eligibility checking
    # ------------------------------------------------------------------

    def check_eligibility(
        self,
        criteria: EligibilityCriteria,
        user: UserProfile
    ) -> EligibilityResult:
        """
        Check if a user profile meets the parsed eligibility criteria.

        Uses deterministic comparisons for age/gender, and keyword
        matching for conditions and medications.

        Args:
            criteria: Parsed eligibility criteria from parse_eligibility()
            user: User's self-reported profile

        Returns:
            EligibilityResult with eligibility status and reasons
        """
        met = []
        unmet = []
        uncertain = []

        # --- Age check ---
        self._check_age(criteria, user, met, unmet, uncertain)

        # --- Gender check ---
        self._check_gender(criteria, user, met, unmet, uncertain)

        # --- Inclusion criteria (conditions the user must have) ---
        self._check_inclusions(criteria, user, met, unmet, uncertain)

        # --- Exclusion criteria (conditions that disqualify) ---
        self._check_exclusions(criteria, user, met, unmet, uncertain)

        # --- Medication checks ---
        self._check_medications(criteria, user, met, unmet, uncertain)

        # --- Determine overall eligibility ---
        # Eligible if no unmet criteria (uncertain ones get a pass with warning)
        is_eligible = len(unmet) == 0

        # Build summary
        summary = self._build_summary(is_eligible, met, unmet, uncertain)

        return EligibilityResult(
            is_eligible=is_eligible,
            met_criteria=met,
            unmet_criteria=unmet,
            uncertain_criteria=uncertain,
            summary=summary
        )

    # ------------------------------------------------------------------
    # LLM-based parsing
    # ------------------------------------------------------------------

    def _parse_with_gemini(self, raw_text: str) -> EligibilityCriteria:
        """Send eligibility text to Gemini and parse the structured response."""
        if self._model is None:
            raise RuntimeError("Gemini model not available (missing API key or SDK)")

        prompt = EXTRACTION_PROMPT.format(eligibility_text=raw_text)

        response = self._model.generate_content(prompt)
        response_text = response.text.strip()

        # Clean potential markdown fencing
        response_text = self._clean_json_response(response_text)

        # Parse JSON
        parsed = json.loads(response_text)

        # Convert to dataclass
        return EligibilityCriteria(
            min_age=parsed.get("min_age"),
            max_age=parsed.get("max_age"),
            gender=Gender(parsed.get("gender", "all").lower()),
            inclusion_criteria=parsed.get("inclusion_criteria", []),
            exclusion_criteria=parsed.get("exclusion_criteria", []),
            required_medications=parsed.get("required_medications", []),
            excluded_medications=parsed.get("excluded_medications", []),
            other_requirements=parsed.get("other_requirements", []),
            raw_text=raw_text,
            parse_confidence=parsed.get("parse_confidence", 0.8)
        )

    # ------------------------------------------------------------------
    # Regex fallback parsing
    # ------------------------------------------------------------------

    def _parse_with_regex(self, raw_text: str) -> EligibilityCriteria:
        """
        Basic regex-based extraction for age and gender.
        Used as fallback when Gemini is unavailable.
        """
        min_age, max_age = self._extract_age_range(raw_text)
        gender = self._extract_gender(raw_text)
        inclusions, exclusions = self._extract_criteria_sections(raw_text)

        return EligibilityCriteria(
            min_age=min_age,
            max_age=max_age,
            gender=gender,
            inclusion_criteria=inclusions,
            exclusion_criteria=exclusions,
            raw_text=raw_text,
            parse_confidence=0.5  # lower confidence for regex
        )

    def _extract_age_range(self, text: str) -> tuple[int | None, int | None]:
        """Extract min/max age from eligibility text using regex."""
        min_age = None
        max_age = None

        # Pattern: "aged 18-65", "ages 18 to 65", "18-65 years"
        range_pattern = r'(?:aged?|ages?)\s*(\d{1,3})\s*(?:-|to)\s*(\d{1,3})'
        match = re.search(range_pattern, text, re.IGNORECASE)
        if match:
            min_age = int(match.group(1))
            max_age = int(match.group(2))
            return min_age, max_age

        # Pattern: "≥18 years" or ">= 18 years" or "at least 18"
        min_patterns = [
            r'(?:≥|>=)\s*(\d{1,3})\s*(?:years?|yrs?)',
            r'at\s+least\s+(\d{1,3})\s*(?:years?|yrs?)',
            r'minimum\s+age[:\s]*(\d{1,3})',
            r'(\d{1,3})\s*(?:years?|yrs?)\s*(?:or|and)\s*older',
        ]
        for pattern in min_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                min_age = int(match.group(1))
                break

        # Pattern: "≤65 years" or "<= 65" or "up to 65"
        max_patterns = [
            r'(?:≤|<=)\s*(\d{1,3})\s*(?:years?|yrs?)',
            r'up\s+to\s+(\d{1,3})\s*(?:years?|yrs?)',
            r'maximum\s+age[:\s]*(\d{1,3})',
            r'(\d{1,3})\s*(?:years?|yrs?)\s*(?:or|and)\s*younger',
        ]
        for pattern in max_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                max_age = int(match.group(1))
                break

        return min_age, max_age

    def _extract_gender(self, text: str) -> Gender:
        """Extract gender requirement from text using word-boundary matching."""
        text_lower = text.lower()

        # Use regex word boundaries to prevent "female" from matching "male"
        male_patterns = [
            r'\bmale\s+only\b', r'\bmales\s+only\b',
            r'\bmen\s+only\b', r'\bmale\s+participants\s+only\b',
        ]
        female_patterns = [
            r'\bfemale\s+only\b', r'\bfemales\s+only\b',
            r'\bwomen\s+only\b', r'\bfemale\s+participants\s+only\b',
            r'\bpregnant\s+women\b',
        ]

        male_only = any(re.search(p, text_lower) for p in male_patterns)
        female_only = any(re.search(p, text_lower) for p in female_patterns)

        if male_only and not female_only:
            return Gender.MALE
        elif female_only and not male_only:
            return Gender.FEMALE
        return Gender.ALL

    def _extract_criteria_sections(self, text: str) -> tuple[list[str], list[str]]:
        """Split text into inclusion and exclusion criteria lists."""
        inclusions = []
        exclusions = []

        # Try to split on "Inclusion Criteria" / "Exclusion Criteria" headers
        parts = re.split(
            r'(?i)(inclusion\s+criteria|exclusion\s+criteria)\s*:?\s*',
            text
        )

        current_section = None
        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                continue
            if re.match(r'(?i)inclusion\s+criteria', part_stripped):
                current_section = "inclusion"
            elif re.match(r'(?i)exclusion\s+criteria', part_stripped):
                current_section = "exclusion"
            elif current_section:
                # Split on bullet points, numbered items, or newlines
                items = re.split(r'\n\s*[-•*\d.]+\s*', part_stripped)
                items = [item.strip() for item in items if item.strip()]
                if current_section == "inclusion":
                    inclusions.extend(items)
                else:
                    exclusions.extend(items)

        return inclusions, exclusions

    # ------------------------------------------------------------------
    # Eligibility check helpers
    # ------------------------------------------------------------------

    def _check_age(self, criteria, user, met, unmet, uncertain):
        """Check if user's age falls within the trial's range."""
        if user.age is None:
            if criteria.min_age is not None or criteria.max_age is not None:
                age_range = self._format_age_range(criteria.min_age, criteria.max_age)
                uncertain.append(f"Age not provided — trial requires {age_range}")
            return

        if criteria.min_age is not None and user.age < criteria.min_age:
            unmet.append(
                f"Age {user.age} is below minimum age {criteria.min_age}"
            )
            return

        if criteria.max_age is not None and user.age > criteria.max_age:
            unmet.append(
                f"Age {user.age} is above maximum age {criteria.max_age}"
            )
            return

        age_range = self._format_age_range(criteria.min_age, criteria.max_age)
        if age_range:
            met.append(f"Age {user.age} is within required range ({age_range})")
        else:
            met.append("No age restriction for this trial")

    def _check_gender(self, criteria, user, met, unmet, uncertain):
        """Check if user's gender matches the trial's requirement."""
        if criteria.gender == Gender.ALL:
            met.append("No gender restriction")
            return

        if user.gender is None:
            uncertain.append(
                f"Gender not provided — trial requires {criteria.gender.value}"
            )
            return

        if user.gender.lower() == criteria.gender.value:
            met.append(f"Gender ({user.gender}) matches requirement")
        else:
            unmet.append(
                f"Trial requires {criteria.gender.value}, "
                f"user is {user.gender}"
            )

    def _check_inclusions(self, criteria, user, met, unmet, uncertain):
        """Check if user has the required conditions (inclusion criteria)."""
        if not criteria.inclusion_criteria:
            return

        user_conditions_lower = [c.lower() for c in user.conditions]

        for criterion in criteria.inclusion_criteria:
            criterion_lower = criterion.lower()

            # Check if any user condition matches this criterion (substring match)
            matched = any(
                self._terms_match(cond, criterion_lower)
                for cond in user_conditions_lower
            )

            if matched:
                met.append(f"Meets inclusion: {criterion}")
            elif user.conditions:
                # User gave conditions but none matched — might still qualify
                # depending on how specific the criterion is
                uncertain.append(f"Unclear if meets inclusion: {criterion}")
            else:
                uncertain.append(f"No conditions provided to check: {criterion}")

    def _check_exclusions(self, criteria, user, met, unmet, uncertain):
        """Check if user has any disqualifying conditions."""
        if not criteria.exclusion_criteria:
            return

        user_conditions_lower = [c.lower() for c in user.conditions]

        for criterion in criteria.exclusion_criteria:
            criterion_lower = criterion.lower()

            # Check if any user condition triggers this exclusion
            matched = any(
                self._terms_match(cond, criterion_lower)
                for cond in user_conditions_lower
            )

            if matched:
                unmet.append(f"Excluded by: {criterion}")
            else:
                met.append(f"Not excluded by: {criterion}")

    def _check_medications(self, criteria, user, met, unmet, uncertain):
        """Check medication requirements and exclusions."""
        user_meds_lower = [m.lower() for m in user.medications]

        # Required medications
        for med in criteria.required_medications:
            med_lower = med.lower()
            if any(self._terms_match(um, med_lower) for um in user_meds_lower):
                met.append(f"Currently taking required medication: {med}")
            elif user.medications:
                uncertain.append(f"Not confirmed on required medication: {med}")
            else:
                uncertain.append(f"Medication info not provided — requires: {med}")

        # Excluded medications
        for med in criteria.excluded_medications:
            med_lower = med.lower()
            if any(self._terms_match(um, med_lower) for um in user_meds_lower):
                unmet.append(f"Currently taking excluded medication: {med}")
            else:
                met.append(f"Not taking excluded medication: {med}")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _terms_match(user_term: str, criterion_term: str) -> bool:
        """
        Check if a user's term matches a criterion term.
        Uses substring matching in both directions for flexibility.
        """
        return user_term in criterion_term or criterion_term in user_term

    @staticmethod
    def _format_age_range(min_age: int | None, max_age: int | None) -> str:
        """Format age range for display."""
        if min_age is not None and max_age is not None:
            return f"{min_age}-{max_age} years"
        elif min_age is not None:
            return f"{min_age}+ years"
        elif max_age is not None:
            return f"up to {max_age} years"
        return ""

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Remove markdown code fences if Gemini wraps the JSON."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def _build_summary(
        is_eligible: bool,
        met: list[str],
        unmet: list[str],
        uncertain: list[str]
    ) -> str:
        """Build a human-readable eligibility summary."""
        if is_eligible and not uncertain:
            return (
                f"You appear to meet all eligibility criteria. "
                f"({len(met)} criteria met)"
            )
        elif is_eligible and uncertain:
            return (
                f"You may be eligible, but {len(uncertain)} criteria "
                f"could not be fully verified from the information provided. "
                f"({len(met)} criteria confirmed)"
            )
        else:
            return (
                f"You may not be eligible for this trial. "
                f"{len(unmet)} criteria not met. "
                f"Please review the details below."
            )

    def __repr__(self) -> str:
        return f"EligibilityParser(model={self._model_name})"