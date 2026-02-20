import logging
from typing import Optional
from dataclasses import dataclass, field

# Internal API layer (already built)
from src.api.client import ClinicalTrialsClient
from src.api.models import Trial
from src.api.exceptions import APIError, RateLimitError

logger = logging.getLogger(__name__)


# ── Interfaces for Member 2's tools (stubs until delivered) ────────────

def _default_term_mapper(term: str) -> str:
    """Passthrough stub. Replaced by Member 2's medical_term_mapper."""
    logger.debug(f"Term mapper stub: '{term}' passed through unchanged")
    return term


# ── Search parameters ──────────────────────────────────────────────────

@dataclass
class SearchParams:
    """Structured container for all search inputs."""
    condition: str
    location: Optional[str] = None
    status: str = "RECRUITING"
    age: Optional[int] = None
    gender: Optional[str] = None
    max_results: int = 20
    sort_by: str = "relevance"  # "relevance" | "date" | "distance"
    page_size: int = 10

    def validate(self):
        if not self.condition or not self.condition.strip():
            raise ValueError("Condition is required and cannot be empty")
        if self.age is not None and (self.age < 0 or self.age > 120):
            raise ValueError(f"Age must be between 0 and 120, got {self.age}")
        if self.gender and self.gender.lower() not in ("male", "female", "all", "any"):
            raise ValueError(f"Gender must be male/female/all, got {self.gender}")
        if self.max_results < 1 or self.max_results > 100:
            raise ValueError(f"max_results must be 1-100, got {self.max_results}")


# ── Search result container ────────────────────────────────────────────

@dataclass
class SearchResult:
    """Wraps search output with metadata for the agent."""
    trials: list[Trial] = field(default_factory=list)
    total_found: int = 0
    query_used: str = ""       # the actual term sent to API (after mapping)
    original_query: str = ""   # what the user typed
    filters_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Main tool class ────────────────────────────────────────────────────

class TrialSearcher:
    """
    High-level trial search tool for the agent to call.

    Usage:
        searcher = TrialSearcher(api_client)
        result = searcher.search(SearchParams(condition="diabetes", location="Boston"))
    """

    def __init__(
        self,
        api_client: ClinicalTrialsClient,
        term_mapper=None,
    ):
        self.client = api_client
        self.map_term = term_mapper or _default_term_mapper

    def search(self, params: SearchParams) -> SearchResult:
        """
        Main entry point. Validates input, maps terms, queries API,
        and returns filtered results.
        """
        params.validate()
        result = SearchResult(original_query=params.condition)

        # Step 1: Map lay terms → medical terms
        mapped_condition = self._map_condition(params.condition)
        result.query_used = mapped_condition

        # Step 2: Query the API
        trials = self._fetch_trials(mapped_condition, params, result)
        if not trials:
            return result

        # Step 3: Post-process
        trials = self._filter_by_status(trials, params.status, result)
        trials = self._filter_by_demographics(trials, params, result)
        trials = self._sort(trials, params.sort_by)
        trials = trials[: params.max_results]

        result.trials = trials
        result.total_found = len(trials)
        return result

    # ── Internal steps ─────────────────────────────────────────────

    def _map_condition(self, condition: str) -> str:
        """Use the medical term mapper (Member 2) to normalize the term."""
        try:
            mapped = self.map_term(condition)
            if mapped != condition:
                logger.info(f"Mapped '{condition}' → '{mapped}'")
            return mapped
        except Exception as e:
            logger.warning(f"Term mapping failed for '{condition}': {e}. Using original.")
            return condition

    def _fetch_trials(
        self, condition: str, params: SearchParams, result: SearchResult
    ) -> list[Trial]:
        """Call the API client and handle errors gracefully."""
        try:
            trials = self.client.search_studies(
                condition=condition,
                location=params.location,
                status=params.status,
                page_size=params.page_size,
            )
            logger.info(f"API returned {len(trials)} trials for '{condition}'")
            return trials

        except RateLimitError:
            msg = "Rate limit reached. Try again in a minute."
            logger.warning(msg)
            result.errors.append(msg)
            return []

        except APIError as e:
            msg = f"API error during search: {e}"
            logger.error(msg)
            result.errors.append(msg)
            return []

        except Exception as e:
            msg = f"Unexpected error during search: {e}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)
            return []

    def _filter_by_status(
        self, trials: list[Trial], status: str, result: SearchResult
    ) -> list[Trial]:
        """Double-check status filter (API should handle this, but verify)."""
        if not status:
            return trials
        filtered = [t for t in trials if t.status and t.status.upper() == status.upper()]
        if len(filtered) < len(trials):
            dropped = len(trials) - len(filtered)
            logger.debug(f"Status filter dropped {dropped} trials")
            result.filters_applied.append(f"status={status}")
        return filtered

    def _filter_by_demographics(
        self, trials: list[Trial], params: SearchParams, result: SearchResult
    ) -> list[Trial]:
        """
        Basic demographic pre-filter using eligibility metadata.
        The detailed eligibility check (Member 2's parser) happens later
        in the agent pipeline — this is just a quick pass.
        """
        filtered = trials

        if params.age is not None:
            before = len(filtered)
            filtered = [t for t in filtered if self._age_in_range(t, params.age)]
            if len(filtered) < before:
                result.filters_applied.append(f"age={params.age}")

        if params.gender and params.gender.lower() not in ("all", "any"):
            before = len(filtered)
            filtered = [t for t in filtered if self._gender_matches(t, params.gender)]
            if len(filtered) < before:
                result.filters_applied.append(f"gender={params.gender}")

        return filtered

    def _sort(self, trials: list[Trial], sort_by: str) -> list[Trial]:
        """Sort trials. Distance sorting happens in geo_matcher (Member 3)."""
        if sort_by == "date":
            return sorted(
                trials,
                key=lambda t: t.last_updated or "",
                reverse=True,
            )
        # "relevance" = keep API's original ordering
        # "distance" = handled downstream by geo_matcher
        return trials

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _age_in_range(trial: Trial, age: int) -> bool:
        """Check if age falls within trial's min/max age range."""
        elig = getattr(trial, "eligibility", None)
        if not elig:
            return True  # no info = don't exclude
        min_age = getattr(elig, "min_age", None)
        max_age = getattr(elig, "max_age", None)
        if min_age is not None and age < min_age:
            return False
        if max_age is not None and age > max_age:
            return False
        return True

    @staticmethod
    def _gender_matches(trial: Trial, gender: str) -> bool:
        """Check if trial accepts the user's gender."""
        elig = getattr(trial, "eligibility", None)
        if not elig:
            return True
        trial_gender = getattr(elig, "gender", "all")
        if not trial_gender or trial_gender.lower() == "all":
            return True
        return trial_gender.lower() == gender.lower()