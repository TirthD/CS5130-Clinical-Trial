import re
import logging
from dataclasses import dataclass, field

from src.agent.prompts import DISCLAIMER, MEDICAL_ADVICE_REDIRECT

logger = logging.getLogger(__name__)


# ── Patterns and Keywords ──────────────────────────────────────────────

# Phrases that suggest the agent is giving medical advice
MEDICAL_ADVICE_PATTERNS = [
    r"\byou should (take|start|stop|try|consider taking|ask about)\b",
    r"\bi (recommend|suggest|advise) (you|that you)\b",
    r"\b(best treatment|right treatment|correct treatment) for you\b",
    r"\byou (need|must) (take|start|get|begin)\b",
    r"\bthis (drug|medication|treatment) (is|would be) (best|right|ideal) for\b",
    r"\b(stop|discontinue|change) your (medication|treatment|dosage)\b",
    r"\byou (are|might be) (suffering from|diagnosed with)\b",
    r"\bbased on your symptoms.{0,20}(likely|probably|could be)\b",
    r"\byou (likely|probably) have\b",
]

# Valid NCT ID format
NCT_PATTERN = re.compile(r"NCT\d{8}")

# Words that hint the agent may be fabricating details
HEDGING_BEFORE_DATA = [
    r"\b(i think|i believe|probably|likely|might be|could be)\b.{0,30}\bNCT",
    r"\bNCT\d{8}\b.{0,30}\b(i think|probably|i'm not sure)\b",
]


# ── Safety Check Results ───────────────────────────────────────────────

@dataclass
class SafetyCheckResult:
    """Outcome of running all safety checks on a response."""
    is_safe: bool = True
    issues: list[str] = field(default_factory=list)
    modified_response: str | None = None  # set if we auto-fixed something

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_safe = False
        logger.warning(f"Safety issue: {issue}")


# ── Main Safety Checker ───────────────────────────────────────────────

class SafetyGuard:
    """
    Runs all safety checks on an agent response.

    Usage:
        guard = SafetyGuard(known_nct_ids={"NCT12345678", "NCT87654321"})
        result = guard.check(response_text)
        if not result.is_safe:
            # handle or use result.modified_response
    """

    def __init__(self, known_nct_ids: set[str] | None = None):
        """
        Args:
            known_nct_ids: NCT IDs returned by the API during this query.
                Used to detect hallucinated IDs in the response.
        """
        self.known_nct_ids = known_nct_ids or set()
        self._advice_patterns = [re.compile(p, re.IGNORECASE) for p in MEDICAL_ADVICE_PATTERNS]
        self._hedging_patterns = [re.compile(p, re.IGNORECASE) for p in HEDGING_BEFORE_DATA]

    def check(self, response: str) -> SafetyCheckResult:
        """
        Run all safety checks on the agent's response.
        Returns a SafetyCheckResult with issues found and
        optionally a modified (fixed) response.
        """
        result = SafetyCheckResult()
        modified = response

        # Check 1: Medical advice detection
        self._check_medical_advice(response, result)

        # Check 2: NCT citation validation
        self._check_nct_citations(response, result)

        # Check 3: Hallucination signals
        self._check_hallucination_signals(response, result)

        # Check 4: Disclaimer present (auto-fix if missing)
        modified = self._ensure_disclaimer(modified, result)

        result.modified_response = modified
        return result

    # ── Individual Checks ──────────────────────────────────────────

    def _check_medical_advice(self, response: str, result: SafetyCheckResult):
        """Detect language that sounds like medical advice."""
        for pattern in self._advice_patterns:
            match = pattern.search(response)
            if match:
                result.add_issue(
                    f"Potential medical advice detected: '{match.group()}'"
                )
                # Don't break — log all matches for debugging

    def _check_nct_citations(self, response: str, result: SafetyCheckResult):
        """
        Verify that any NCT IDs in the response were actually
        returned by the API (not hallucinated).
        """
        mentioned_ids = set(NCT_PATTERN.findall(response))

        if not mentioned_ids:
            # No trials mentioned — fine for general responses
            return

        if not self.known_nct_ids:
            # No known IDs to compare against — can't verify
            result.add_issue(
                "Response mentions NCT IDs but no known IDs were provided "
                "for validation. Cannot verify authenticity."
            )
            return

        hallucinated = mentioned_ids - self.known_nct_ids
        if hallucinated:
            result.add_issue(
                f"Potentially hallucinated NCT IDs: {hallucinated}. "
                "These were not returned by the API."
            )

    def _check_hallucination_signals(self, response: str, result: SafetyCheckResult):
        """Detect hedging language near trial data, which may signal fabrication."""
        for pattern in self._hedging_patterns:
            match = pattern.search(response)
            if match:
                result.add_issue(
                    f"Uncertain language near trial data: '{match.group()}'. "
                    "Agent may be fabricating details."
                )

    def _ensure_disclaimer(self, response: str, result: SafetyCheckResult) -> str:
        """Check for disclaimer and append if missing."""
        # Check if disclaimer (or a substantial part of it) is present
        disclaimer_key_phrase = "NOT medical advice"
        if disclaimer_key_phrase in response:
            return response

        result.add_issue("Disclaimer was missing — auto-appended.")
        return response.rstrip() + "\n\n" + DISCLAIMER


# ── Standalone Utility Functions ───────────────────────────────────────

def is_medical_advice_request(user_query: str) -> bool:
    """
    Quick check on the user's INPUT to detect if they're asking
    for medical advice (so the agent can redirect early).
    """
    advice_signals = [
        r"\bshould i (take|start|stop|try|switch)\b",
        r"\bwhat (treatment|medication|drug) (should|do you recommend)\b",
        r"\bis .{1,40} (safe|dangerous|effective) for me\b",
        r"\bwhat('s| is) (wrong with me|my diagnosis)\b",
        r"\bcan you (diagnose|prescribe|recommend a treatment)\b",
        r"\b(treat|cure|fix) my\b",
        r"\bam i (sick|ill|dying)\b",
    ]
    for pattern in advice_signals:
        if re.search(pattern, user_query, re.IGNORECASE):
            return True
    return False


def get_medical_advice_redirect() -> str:
    """Return the standard redirect message for medical advice requests."""
    return MEDICAL_ADVICE_REDIRECT


def sanitize_response(response: str, known_nct_ids: set[str] | None = None) -> str:
    """
    Convenience function: run safety checks and return the
    cleaned response. For quick use without the full SafetyGuard.
    """
    guard = SafetyGuard(known_nct_ids=known_nct_ids)
    result = guard.check(response)

    if result.issues:
        logger.info(f"Safety check found {len(result.issues)} issue(s)")
        for issue in result.issues:
            logger.info(f"  - {issue}")

    return result.modified_response or response