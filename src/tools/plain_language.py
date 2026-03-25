"""
Plain Language Translator Tool
===============================
Translates complex medical/clinical trial text into simple, easy-to-understand
English using Google Gemini.

Used at the end of the pipeline to make trial descriptions, eligibility
criteria, and procedures accessible to non-medical users.

Usage:
    translator = PlainLanguageTranslator()

    # Translate a single text
    result = translator.translate(complex_medical_text)
    print(result.plain_text)

    # Translate multiple texts in one call (saves API calls)
    results = translator.translate_batch([text1, text2, text3])

    # Translate a full trial summary
    summary = translator.translate_trial_summary(trial_data)
"""

from __future__ import annotations

import json
import os
import logging
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    """Type of medical content being translated — affects prompt strategy."""
    TRIAL_DESCRIPTION = "trial_description"
    ELIGIBILITY = "eligibility"
    PROCEDURE = "procedure"
    GENERAL = "general"


@dataclass
class TranslationResult:
    """Result of translating a single piece of medical text."""
    original_text: str            # The raw medical text
    plain_text: str               # Simplified version
    content_type: ContentType     # What kind of content was translated
    definitions: list[str] = field(default_factory=list)  # Key terms defined
    success: bool = True          # Whether translation succeeded
    source: str = "gemini"        # "gemini" or "fallback"


@dataclass
class TrialSummary:
    """A fully translated clinical trial summary in plain language."""
    title: str = ""
    purpose: str = ""
    what_happens: str = ""
    who_can_join: str = ""
    location: str = ""
    contact: str = ""
    nct_id: str = ""
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPTS = {
    ContentType.TRIAL_DESCRIPTION: """You are a medical translator who explains clinical trials to everyday people.

Rewrite the following clinical trial description so that someone with NO medical background can understand it.

Rules:
- Use short, simple sentences (8th grade reading level)
- Define any medical terms in parentheses the first time they appear
- Keep it accurate — do NOT add information that is not in the original
- Do NOT give medical advice or opinions
- Keep the same overall meaning and structure
- Be warm and clear, not robotic

Text to simplify:
\"\"\"
{text}
\"\"\"

Return ONLY the simplified text. No preamble, no extra commentary.""",

    ContentType.ELIGIBILITY: """You are a medical translator who helps people understand clinical trial eligibility requirements.

Rewrite the following eligibility criteria in plain, simple English that anyone can understand.

Rules:
- Turn each criterion into a clear yes/no question or simple statement
- Define medical terms in parentheses, e.g. "HbA1c (a blood sugar test)"
- Use "you" language — e.g. "You must be between 18 and 65 years old"
- Keep every requirement — do NOT skip any criteria
- Do NOT give medical advice
- Clearly separate "Who CAN join" and "Who CANNOT join"

Eligibility text:
\"\"\"
{text}
\"\"\"

Return ONLY the simplified text. No preamble, no extra commentary.""",

    ContentType.PROCEDURE: """You are a medical translator who explains clinical trial procedures to patients.

Rewrite the following trial procedures/interventions so a non-medical person understands what will happen to them.

Rules:
- Explain what happens step by step in simple terms
- Define medical procedures in plain language, e.g. "biopsy (a small tissue sample taken with a needle)"
- Use "you" language — e.g. "You will visit the clinic every 2 weeks"
- Be honest about what might be uncomfortable but keep a reassuring tone
- Do NOT add information that is not in the original
- Do NOT give medical advice

Procedure text:
\"\"\"
{text}
\"\"\"

Return ONLY the simplified text. No preamble, no extra commentary.""",

    ContentType.GENERAL: """You are a medical translator who makes complex health information easy to understand.

Rewrite the following medical text in plain, simple English.

Rules:
- Use short sentences (8th grade reading level)
- Define medical terms in parentheses the first time they appear
- Keep it accurate — do NOT change the meaning
- Do NOT give medical advice or opinions

Text to simplify:
\"\"\"
{text}
\"\"\"

Return ONLY the simplified text. No preamble, no extra commentary.""",
}

TRIAL_SUMMARY_PROMPT = """You are a medical translator. Summarize this clinical trial in plain English that anyone can understand.

Return ONLY valid JSON (no markdown, no backticks) with this structure:
{{
    "title": "A simple, clear title for the study",
    "purpose": "What this study is trying to find out (1-2 sentences)",
    "what_happens": "What participants will experience (2-3 sentences)",
    "who_can_join": "Key requirements in simple terms (2-3 bullet points as a single string)",
    "definitions": ["term1: simple definition", "term2: simple definition"]
}}

Trial information:
\"\"\"
Title: {title}
Description: {description}
Eligibility: {eligibility}
Interventions: {interventions}
\"\"\"
"""

DEFINITIONS_PROMPT = """Extract and define the medical terms from this text. For each term, give a brief plain-English definition.

Return ONLY valid JSON — a list of strings in the format "term: definition".
Example: ["HbA1c: a blood test that shows your average blood sugar over 3 months", "placebo: a dummy treatment with no active medicine"]

Text:
\"\"\"
{text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Main translator class
# ---------------------------------------------------------------------------

class PlainLanguageTranslator:
    """
    Translates medical and clinical trial text into plain English
    using Google Gemini.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        """
        Initialize the translator with Gemini credentials.

        Args:
            api_key: Google Gemini API key. Falls back to config.py / env var.
            model_name: Gemini model to use. Defaults to config setting.
        """
        # Load .env from project root
        try:
            from dotenv import load_dotenv
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            dotenv_path = os.path.join(project_root, ".env")
            load_dotenv(dotenv_path, override=True)
        except ImportError:
            pass

        # Load config
        try:
            import config
            self._api_key = api_key or config.GEMINI_API_KEY
            self._model_name = model_name or config.GEMINI_MODEL
        except ImportError:
            self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            self._model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # Simple in-memory cache: hash(input) -> TranslationResult
        self._cache: dict[str, TranslationResult] = {}

        if not self._api_key:
            logger.warning(
                "No Gemini API key provided. "
                "PlainLanguageTranslator will use basic fallback."
            )
            self._model = None
            return

        # Initialize Gemini
        try:
            import google.generativeai as _genai
            _genai.configure(api_key=self._api_key)
            self._model = _genai.GenerativeModel(self._model_name)
        except ImportError:
            logger.warning(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
            self._model = None

    # ------------------------------------------------------------------
    # Main translation methods
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        content_type: ContentType = ContentType.GENERAL,
        use_cache: bool = True,
    ) -> TranslationResult:
        """
        Translate a piece of medical text into plain English.

        Args:
            text: The medical text to simplify.
            content_type: Type of content (affects prompt strategy).
            use_cache: Whether to check/store in cache.

        Returns:
            TranslationResult with the simplified text.
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                plain_text="",
                content_type=content_type,
                success=False,
                source="none"
            )

        # Check cache
        if use_cache:
            cache_key = self._make_cache_key(text, content_type)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {content_type.value} translation")
                return self._cache[cache_key]

        # Try Gemini
        try:
            result = self._translate_with_gemini(text, content_type)
            if use_cache:
                self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Gemini translation failed: {e}. Using fallback.")

        # Fallback
        result = self._translate_fallback(text, content_type)
        if use_cache:
            self._cache[cache_key] = result
        return result

    def translate_batch(
        self,
        texts: list[str],
        content_type: ContentType = ContentType.GENERAL,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts. Each is translated individually
        but benefits from caching.

        Args:
            texts: List of medical texts to simplify.
            content_type: Type of content for all texts.

        Returns:
            List of TranslationResults in same order as input.
        """
        return [self.translate(text, content_type) for text in texts]

    def translate_trial_summary(
        self,
        title: str = "",
        description: str = "",
        eligibility: str = "",
        interventions: str = "",
        location: str = "",
        contact: str = "",
        nct_id: str = "",
    ) -> TrialSummary:
        """
        Translate a complete clinical trial into a plain-language summary.

        Args:
            title: Trial official title.
            description: Trial description/brief summary.
            eligibility: Raw eligibility criteria text.
            interventions: Interventions/procedures text.
            location: Trial location(s).
            contact: Contact information.
            nct_id: ClinicalTrials.gov NCT ID.

        Returns:
            TrialSummary with all fields in plain English.
        """
        try:
            from config import MEDICAL_DISCLAIMER
        except ImportError:
            MEDICAL_DISCLAIMER = (
                "This is for informational purposes only. "
                "Always consult a healthcare provider before enrolling in a trial."
            )

        # Try Gemini-powered full summary
        try:
            summary = self._summarize_with_gemini(
                title, description, eligibility, interventions
            )
            summary.location = location
            summary.contact = contact
            summary.nct_id = nct_id
            summary.disclaimer = MEDICAL_DISCLAIMER
            return summary
        except Exception as e:
            logger.warning(f"Gemini summary failed: {e}. Building from parts.")

        # Fallback: translate each piece individually
        return self._build_summary_from_parts(
            title, description, eligibility, interventions,
            location, contact, nct_id, MEDICAL_DISCLAIMER
        )

    def extract_definitions(self, text: str) -> list[str]:
        """
        Extract medical terms from text and define them in plain English.

        Args:
            text: Medical text to extract terms from.

        Returns:
            List of "term: definition" strings.
        """
        if not self._model or not text.strip():
            return []

        try:
            prompt = DEFINITIONS_PROMPT.format(text=text)
            response = self._call_gemini_with_retry(prompt)
            response_text = self._clean_json_response(response.text.strip())
            definitions = json.loads(response_text)
            if isinstance(definitions, list):
                return definitions
        except Exception as e:
            logger.warning(f"Definition extraction failed: {e}")

        return []

    # ------------------------------------------------------------------
    # Gemini translation
    # ------------------------------------------------------------------

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3):
        """
        Call Gemini with automatic retry on rate limit (429) errors.

        Waits and retries when hitting the free tier quota limit,
        so tests and real usage don't fail on transient rate limits.

        Args:
            prompt: The prompt to send to Gemini.
            max_retries: Maximum number of retry attempts.

        Returns:
            Gemini response object.
        """
        if self._model is None:
            raise RuntimeError("Gemini model not available")

        for attempt in range(max_retries + 1):
            try:
                return self._model.generate_content(prompt)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries:
                        wait_time = 60  # free tier resets per minute
                        logger.info(
                            f"Rate limited. Waiting {wait_time}s before retry "
                            f"({attempt + 1}/{max_retries})..."
                        )
                        print(
                            f"  ⏳ Rate limited — waiting {wait_time}s "
                            f"(retry {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

    def _translate_with_gemini(
        self, text: str, content_type: ContentType
    ) -> TranslationResult:
        """Send text to Gemini with the appropriate prompt template."""
        if self._model is None:
            raise RuntimeError("Gemini model not available")

        prompt_template = PROMPTS.get(content_type, PROMPTS[ContentType.GENERAL])
        prompt = prompt_template.format(text=text)

        response = self._call_gemini_with_retry(prompt)
        plain_text = response.text.strip()

        return TranslationResult(
            original_text=text,
            plain_text=plain_text,
            content_type=content_type,
            definitions=[],  # call extract_definitions() separately if needed
            success=True,
            source="gemini"
        )

    def _summarize_with_gemini(
        self,
        title: str,
        description: str,
        eligibility: str,
        interventions: str,
    ) -> TrialSummary:
        """Generate a complete plain-language trial summary with Gemini."""
        if self._model is None:
            raise RuntimeError("Gemini model not available")

        prompt = TRIAL_SUMMARY_PROMPT.format(
            title=title or "Not provided",
            description=description or "Not provided",
            eligibility=eligibility or "Not provided",
            interventions=interventions or "Not provided",
        )

        response = self._call_gemini_with_retry(prompt)
        response_text = self._clean_json_response(response.text.strip())
        parsed = json.loads(response_text)

        return TrialSummary(
            title=parsed.get("title", title),
            purpose=parsed.get("purpose", ""),
            what_happens=parsed.get("what_happens", ""),
            who_can_join=parsed.get("who_can_join", ""),
        )

    # ------------------------------------------------------------------
    # Fallback translation (no LLM)
    # ------------------------------------------------------------------

    def _translate_fallback(
        self, text: str, content_type: ContentType
    ) -> TranslationResult:
        """
        Basic fallback when Gemini is unavailable.
        Applies simple substitutions for common medical terms.
        """
        SUBSTITUTIONS = {
            "randomized": "randomly assigned",
            "double-blind": "neither patient nor doctor knows who gets the real treatment",
            "placebo-controlled": "compared against a dummy treatment",
            "placebo": "dummy treatment (no active medicine)",
            "efficacy": "how well the treatment works",
            "adverse events": "side effects",
            "contraindicated": "not recommended",
            "etiology": "cause",
            "pathogenesis": "how the disease develops",
            "prognosis": "expected outcome",
            "comorbidity": "other health condition",
            "comorbidities": "other health conditions",
            "myocardial infarction": "heart attack",
            "hypertension": "high blood pressure",
            "hypotension": "low blood pressure",
            "dyspnea": "shortness of breath",
            "edema": "swelling",
            "hemorrhage": "bleeding",
            "renal": "kidney",
            "hepatic": "liver",
            "pulmonary": "lung",
            "cerebrovascular": "brain blood vessel",
            "subcutaneous": "under the skin",
            "intravenous": "into a vein (IV)",
            "hba1c": "HbA1c (a blood sugar test)",
            "egfr": "eGFR (a kidney function test)",
            "bmi": "BMI (body mass index)",
        }

        result_text = text
        for medical_term, plain_term in SUBSTITUTIONS.items():
            # Case-insensitive replacement, preserving surrounding text
            import re
            pattern = re.compile(re.escape(medical_term), re.IGNORECASE)
            result_text = pattern.sub(plain_term, result_text)

        return TranslationResult(
            original_text=text,
            plain_text=result_text,
            content_type=content_type,
            definitions=[],
            success=True,
            source="fallback"
        )

    def _build_summary_from_parts(
        self,
        title: str,
        description: str,
        eligibility: str,
        interventions: str,
        location: str,
        contact: str,
        nct_id: str,
        disclaimer: str,
    ) -> TrialSummary:
        """Build a trial summary by translating each part individually."""
        translated_desc = self.translate(
            description, ContentType.TRIAL_DESCRIPTION
        ) if description else None
        translated_elig = self.translate(
            eligibility, ContentType.ELIGIBILITY
        ) if eligibility else None
        translated_proc = self.translate(
            interventions, ContentType.PROCEDURE
        ) if interventions else None

        return TrialSummary(
            title=title,
            purpose=translated_desc.plain_text if translated_desc else "",
            what_happens=translated_proc.plain_text if translated_proc else "",
            who_can_join=translated_elig.plain_text if translated_elig else "",
            location=location,
            contact=contact,
            nct_id=nct_id,
            disclaimer=disclaimer,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(text: str, content_type: ContentType) -> str:
        """Create a cache key from the input text and content type."""
        content = f"{content_type.value}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Remove markdown code fences if present."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def clear_cache(self) -> int:
        """Clear the translation cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cached translations")
        return count

    @property
    def cache_size(self) -> int:
        """Number of translations currently cached."""
        return len(self._cache)

    def __repr__(self) -> str:
        return f"PlainLanguageTranslator(model={self._model_name}, cached={self.cache_size})"