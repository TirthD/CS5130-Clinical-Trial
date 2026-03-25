"""
Medical Term Mapper Tool
========================
Translates everyday language into standardized medical terminology
for effective ClinicalTrials.gov API searches.

Approach:
    1. Exact match against the synonyms dictionary
    2. Reverse lookup (user already typed a medical term)
    3. Fuzzy matching for misspellings and close variants
    4. Pass-through with low confidence if nothing matches

Usage:
    mapper = MedicalTermMapper()
    result = mapper.map_term("heart attack")
    # => MappingResult(preferred="myocardial infarction", confidence=1.0, ...)
"""

import json
import os
import re
import logging
import difflib
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class MappingResult:
    """Result of mapping a single lay term to a medical term."""
    original_term: str               # What the user typed
    preferred_term: str              # Best medical term for API search
    alternatives: list[str] = field(default_factory=list)  # Other valid terms
    confidence: float = 0.0          # 0.0 - 1.0 (1.0 = exact match)
    match_type: str = "none"         # exact | reverse | fuzzy | passthrough

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """Check if this mapping is trustworthy enough to use directly."""
        return self.confidence >= threshold


@dataclass
class MultiMappingResult:
    """Result of mapping multiple terms from a single query."""
    results: list[MappingResult] = field(default_factory=list)
    raw_query: str = ""

    @property
    def all_preferred_terms(self) -> list[str]:
        """Get all preferred terms for API search."""
        return [r.preferred_term for r in self.results]

    @property
    def low_confidence_terms(self) -> list[MappingResult]:
        """Terms the agent should warn the user about."""
        return [r for r in self.results if not r.is_high_confidence()]


# ---------------------------------------------------------------------------
# Main mapper class
# ---------------------------------------------------------------------------

class MedicalTermMapper:
    """
    Maps lay/everyday medical terms to standardized terminology.

    Loads a synonyms dictionary from JSON and provides exact,
    reverse, and fuzzy matching strategies.
    """

    # Default path relative to project root
    DEFAULT_SYNONYMS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "medical_synonyms.json"
    )

    # Fuzzy match threshold (0-100). Below this, we treat it as no match.
    FUZZY_THRESHOLD = 75

    def __init__(self, synonyms_path: str | None = None):
        """
        Initialize the mapper and load the synonyms dictionary.

        Args:
            synonyms_path: Path to medical_synonyms.json.
                           Defaults to data/medical_synonyms.json in project root.
        """
        self._synonyms_path = synonyms_path or self.DEFAULT_SYNONYMS_PATH
        self._synonyms: dict = {}
        self._reverse_index: dict[str, str] = {}  # medical_term -> lay_term key
        self._load_synonyms()

    # ------------------------------------------------------------------
    # Loading and indexing
    # ------------------------------------------------------------------

    def _load_synonyms(self) -> None:
        """Load the synonyms JSON and build the reverse index."""
        try:
            with open(self._synonyms_path, "r", encoding="utf-8") as f:
                self._synonyms = json.load(f)
            logger.info(f"Loaded {len(self._synonyms)} synonym entries from {self._synonyms_path}")
        except FileNotFoundError:
            logger.error(f"Synonyms file not found: {self._synonyms_path}")
            raise FileNotFoundError(
                f"Medical synonyms file not found at {self._synonyms_path}. "
                "Please ensure data/medical_synonyms.json exists."
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in synonyms file: {e}")
            raise

        # Build reverse index: maps every known medical term back to its lay-term key
        # This lets us recognize when the user already typed a proper medical term
        self._reverse_index = {}
        for lay_term, entry in self._synonyms.items():
            preferred = entry["preferred"].lower()
            self._reverse_index[preferred] = lay_term

            for alt in entry.get("alternatives", []):
                self._reverse_index[alt.lower()] = lay_term

    # ------------------------------------------------------------------
    # Core mapping logic
    # ------------------------------------------------------------------

    def map_term(self, user_term: str) -> MappingResult:
        """
        Map a single user term to a standardized medical term.

        Strategy:
            1. Exact match   — user term is a key in synonyms dict
            2. Reverse match — user already typed a medical/alternative term
            3. Fuzzy match   — handles misspellings and close variants
            4. Pass-through  — return original with low confidence

        Args:
            user_term: The raw term the user typed (e.g. "heart attack")

        Returns:
            MappingResult with the best medical term and confidence score
        """
        if not user_term or not user_term.strip():
            return MappingResult(
                original_term=user_term,
                preferred_term=user_term,
                confidence=0.0,
                match_type="none"
            )

        cleaned = user_term.strip().lower()

        # --- Strategy 1: Exact match against lay-term keys ---
        result = self._try_exact_match(cleaned)
        if result:
            logger.debug(f"Exact match: '{user_term}' -> '{result.preferred_term}'")
            return result

        # --- Strategy 2: Reverse lookup (user typed a medical term) ---
        result = self._try_reverse_match(cleaned)
        if result:
            logger.debug(f"Reverse match: '{user_term}' -> '{result.preferred_term}'")
            return result

        # --- Strategy 3: Fuzzy match ---
        result = self._try_fuzzy_match(cleaned)
        if result:
            logger.debug(f"Fuzzy match: '{user_term}' -> '{result.preferred_term}' "
                        f"(confidence: {result.confidence:.2f})")
            return result

        # --- Strategy 4: Pass-through (no match found) ---
        logger.info(f"No match found for '{user_term}', passing through as-is")
        return MappingResult(
            original_term=user_term,
            preferred_term=user_term,  # return original — API might still find it
            alternatives=[],
            confidence=0.2,
            match_type="passthrough"
        )

    def map_multiple_terms(self, terms: list[str]) -> MultiMappingResult:
        """
        Map a list of terms individually.

        Args:
            terms: List of user terms like ["heart attack", "diabetes"]

        Returns:
            MultiMappingResult containing individual MappingResults
        """
        results = [self.map_term(term) for term in terms]
        return MultiMappingResult(
            results=results,
            raw_query=" | ".join(terms)
        )

    def extract_and_map(self, query: str) -> MultiMappingResult:
        """
        Extract medical condition terms from a natural language query
        and map each one.

        Handles inputs like:
            "I have diabetes and high blood pressure"
            "heart attack, stroke"
            "type 2 diabetes"

        Args:
            query: Full natural language query from the user

        Returns:
            MultiMappingResult with all extracted and mapped terms
        """
        terms = self._extract_terms(query)
        result = self.map_multiple_terms(terms)
        result.raw_query = query
        return result

    # ------------------------------------------------------------------
    # Private matching strategies
    # ------------------------------------------------------------------

    def _try_exact_match(self, cleaned_term: str) -> MappingResult | None:
        """Strategy 1: Direct lookup in synonyms dictionary."""
        if cleaned_term in self._synonyms:
            entry = self._synonyms[cleaned_term]
            return MappingResult(
                original_term=cleaned_term,
                preferred_term=entry["preferred"],
                alternatives=entry.get("alternatives", []),
                confidence=1.0,
                match_type="exact"
            )
        return None

    def _try_reverse_match(self, cleaned_term: str) -> MappingResult | None:
        """Strategy 2: Check if user already typed a known medical term."""
        if cleaned_term in self._reverse_index:
            lay_key = self._reverse_index[cleaned_term]
            entry = self._synonyms[lay_key]
            return MappingResult(
                original_term=cleaned_term,
                preferred_term=entry["preferred"],
                alternatives=entry.get("alternatives", []),
                confidence=0.95,  # slightly below exact since it's indirect
                match_type="reverse"
            )
        return None

    def _try_fuzzy_match(self, cleaned_term: str) -> MappingResult | None:
        """Strategy 3: Fuzzy string matching against all known terms using difflib."""
        # Build candidate list: all lay-term keys + all medical terms
        all_candidates = list(self._synonyms.keys()) + list(self._reverse_index.keys())

        if not all_candidates:
            return None

        # Use difflib to find close matches
        # cutoff is 0-1 scale (our threshold / 100)
        close_matches = difflib.get_close_matches(
            cleaned_term,
            all_candidates,
            n=1,
            cutoff=self.FUZZY_THRESHOLD / 100.0
        )

        if not close_matches:
            return None

        best_match = close_matches[0]
        # Calculate similarity score for confidence
        score = difflib.SequenceMatcher(None, cleaned_term, best_match).ratio()

        # Determine which entry this matched
        if best_match in self._synonyms:
            entry = self._synonyms[best_match]
        elif best_match in self._reverse_index:
            lay_key = self._reverse_index[best_match]
            entry = self._synonyms[lay_key]
        else:
            return None

        return MappingResult(
            original_term=cleaned_term,
            preferred_term=entry["preferred"],
            alternatives=entry.get("alternatives", []),
            confidence=score,  # already 0-1 range
            match_type="fuzzy"
        )

    # ------------------------------------------------------------------
    # Term extraction from natural language
    # ------------------------------------------------------------------

    def _extract_terms(self, query: str) -> list[str]:
        """
        Extract individual medical condition terms from a natural query.

        Handles common patterns:
            "I have X and Y"
            "diagnosed with X, Y, and Z"
            "X, Y"
            "X and Y"

        Args:
            query: Raw user query string

        Returns:
            List of extracted term strings
        """
        cleaned = query.lower().strip()

        # Remove common preamble phrases
        remove_phrases = [
            "i have ", "i've been diagnosed with ", "i was diagnosed with ",
            "i suffer from ", "i'm dealing with ", "diagnosed with ",
            "suffering from ", "dealing with ", "living with ",
            "i've got ", "i have been diagnosed with ",
            "my doctor said i have ", "my diagnosis is ",
        ]
        for phrase in remove_phrases:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):]
                break

        # Remove trailing filler
        remove_suffixes = [
            " what trials are available",
            " find me trials",
            " find trials",
            " clinical trials",
        ]
        for suffix in remove_suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break

        # Split on common delimiters: "and", commas, semicolons
        parts = re.split(r',\s*|\s+and\s+|\s*;\s*', cleaned)

        # Clean up each part
        terms = []
        for part in parts:
            term = part.strip().strip(".")
            if term and len(term) > 1:  # skip empty / single-char fragments
                terms.append(term)

        return terms if terms else [query.strip()]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_all_known_terms(self) -> list[str]:
        """Return all lay terms this mapper knows about."""
        return list(self._synonyms.keys())

    def get_entry(self, lay_term: str) -> dict | None:
        """Get the full synonym entry for a lay term."""
        return self._synonyms.get(lay_term.lower())

    def __repr__(self) -> str:
        return f"MedicalTermMapper(entries={len(self._synonyms)})"