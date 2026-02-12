"""
HTTP client for ClinicalTrials.gov API v2.
Handles requests, rate limiting, pagination, and error handling.
"""

import time
import logging
import requests
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CT_API_RATE_LIMIT, CT_API_PAGE_SIZE
from src.api.endpoints import build_search_params, get_studies_url, get_single_study_url
from src.api.models import Trial, SearchResult
from src.api.exceptions import (
    ClinicalTrialsAPIError,
    RateLimitError,
    StudyNotFoundError,
    InvalidParameterError,
)

logger = logging.getLogger(__name__)


class ClinicalTrialsClient:
    """
    Client for interacting with the ClinicalTrials.gov v2 API.

    Usage:
        client = ClinicalTrialsClient()

        # Search for trials
        results = client.search_trials(condition="diabetes", location="Boston")

        # Get a specific trial
        trial = client.get_trial("NCT04267848")
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.session = requests.Session()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Simple rate limiter: track request timestamps
        self._request_times: list[float] = []
        self._rate_limit = CT_API_RATE_LIMIT  # 50 per minute

    def _wait_for_rate_limit(self):
        """Ensures we don't exceed 50 requests per minute."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self._rate_limit:
            # Wait until the oldest request is more than 60 seconds old
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit approaching. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        self._request_times.append(time.time())

    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """
        Make an HTTP GET request with retry logic and rate limiting.

        Returns:
            Parsed JSON response as a dictionary.
        """
        self._wait_for_rate_limit()

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Request (attempt {attempt}): {url}")
                response = self.session.get(url, params=params, timeout=30)

                # Handle HTTP errors
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        wait = self.retry_delay * attempt
                        logger.warning(f"Rate limited (429). Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    raise RateLimitError()

                if response.status_code == 404:
                    raise StudyNotFoundError(url.split("/")[-1])

                if response.status_code == 400:
                    error_msg = response.json().get("message", "Bad request")
                    raise InvalidParameterError(error_msg)

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    logger.warning(f"Request timed out. Retrying ({attempt}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                    continue
                raise ClinicalTrialsAPIError("Request timed out after all retries.")

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries:
                    logger.warning(f"Connection error. Retrying ({attempt}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                    continue
                raise ClinicalTrialsAPIError("Failed to connect to ClinicalTrials.gov API.")

        raise ClinicalTrialsAPIError("Max retries exceeded.")

    def search_trials(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        location: Optional[str] = None,
        sponsor: Optional[str] = None,
        status: Optional[str | list[str]] = None,
        phase: Optional[str | list[str]] = None,
        nct_ids: Optional[list[str]] = None,
        page_size: int = CT_API_PAGE_SIZE,
        page_token: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> SearchResult:
        """
        Search for clinical trials.

        Args:
            condition:    Disease/condition (e.g., "type 2 diabetes")
            intervention: Drug/treatment (e.g., "metformin")
            location:     Location (e.g., "Boston, MA")
            sponsor:      Sponsor name
            status:       Recruitment status (e.g., "RECRUITING")
            phase:        Trial phase (e.g., "PHASE3")
            nct_ids:      Specific NCT IDs
            page_size:    Results per page (default 10)
            page_token:   Pagination token for next page
            sort:         Sort order

        Returns:
            SearchResult with list of Trial objects and pagination info.
        """
        params = build_search_params(
            condition=condition,
            intervention=intervention,
            location=location,
            sponsor=sponsor,
            status=status,
            phase=phase,
            nct_ids=nct_ids,
            page_size=page_size,
            page_token=page_token,
            sort=sort,
        )

        url = get_studies_url()
        data = self._make_request(url, params=params)

        # Parse response
        total_count = data.get("totalCount", 0)
        next_token = data.get("nextPageToken")
        raw_studies = data.get("studies", [])

        trials = []
        for study in raw_studies:
            try:
                trial = Trial.from_api_response(study)
                trials.append(trial)
            except Exception as e:
                logger.warning(f"Failed to parse study: {e}")
                continue

        return SearchResult(
            total_count=total_count,
            trials=trials,
            next_page_token=next_token,
        )

    def get_trial(self, nct_id: str) -> Trial:
        """
        Fetch a single trial by its NCT ID.

        Args:
            nct_id: The NCT identifier (e.g., "NCT04267848")

        Returns:
            A Trial object with full details.
        """
        url = get_single_study_url(nct_id)
        data = self._make_request(url)
        return Trial.from_api_response(data)

    def search_all_pages(
        self,
        max_pages: int = 5,
        **search_kwargs,
    ) -> SearchResult:
        """
        Search and automatically paginate through multiple pages of results.

        Args:
            max_pages:      Maximum number of pages to fetch (safety limit).
            **search_kwargs: All arguments accepted by search_trials().

        Returns:
            Combined SearchResult with all trials from all pages.
        """
        all_trials = []
        page_token = None

        for page in range(max_pages):
            result = self.search_trials(page_token=page_token, **search_kwargs)
            all_trials.extend(result.trials)

            logger.info(
                f"Page {page + 1}: fetched {len(result.trials)} trials "
                f"({len(all_trials)}/{result.total_count} total)"
            )

            if not result.next_page_token:
                break
            page_token = result.next_page_token

        return SearchResult(
            total_count=result.total_count,
            trials=all_trials,
            next_page_token=result.next_page_token,
        )


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = ClinicalTrialsClient()

    print("--- Searching for recruiting diabetes trials in Boston ---\n")
    results = client.search_trials(
        condition="type 2 diabetes",
        location="Boston",
        status="RECRUITING",
        page_size=3,
    )

    print(f"Total found: {results.total_count}\n")

    for trial in results.trials:
        print(f"NCT ID:  {trial.nct_id}")
        print(f"Title:   {trial.brief_title}")
        print(f"Status:  {trial.overall_status}")
        print(f"Phase:   {trial.phase}")
        print(f"Sponsor: {trial.sponsor}")
        if trial.locations:
            loc = trial.locations[0]
            print(f"Location: {loc.facility}, {loc.city}, {loc.state}")
        print("-" * 60)