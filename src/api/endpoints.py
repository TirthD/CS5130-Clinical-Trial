"""
API endpoint definitions and URL/parameter builders for ClinicalTrials.gov v2 API.
"""

from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    CT_API_STUDIES_ENDPOINT,
    CT_API_PAGE_SIZE,
    VALID_STATUSES,
)


def build_search_params(
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
    count_total: bool = True,
) -> dict:
    """
    Build query parameters for the /studies endpoint.

    Args:
        condition:    Disease or condition (e.g., "diabetes", "lung cancer")
        intervention: Treatment name (e.g., "pembrolizumab")
        location:     Location search (e.g., "Boston", "Massachusetts")
        sponsor:      Sponsor/organization name
        status:       Filter by status - single string or list
                      (e.g., "RECRUITING" or ["RECRUITING", "NOT_YET_RECRUITING"])
        phase:        Trial phase (e.g., "PHASE3")
        nct_ids:      List of specific NCT IDs to look up
        page_size:    Number of results per page (default 10, max 100)
        page_token:   Token for fetching the next page of results
        sort:         Sort order (e.g., "LastUpdatePostDate:desc")
        count_total:  Whether to include total count in response

    Returns:
        Dictionary of query parameters ready for requests.get()
    """
    params = {
        "format": "json",
        "pageSize": min(page_size, 100),
        "countTotal": str(count_total).lower(),
    }

    # Search queries
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if location:
        params["query.locn"] = location
    if sponsor:
        params["query.spons"] = sponsor

    # Filters
    if status:
        if isinstance(status, str):
            status = [status]
        # Validate status values
        for s in status:
            if s not in VALID_STATUSES:
                raise ValueError(
                    f"Invalid status '{s}'. Must be one of: {VALID_STATUSES}"
                )
        params["filter.overallStatus"] = "|".join(status)

    if phase:
        if isinstance(phase, str):
            phase = [phase]
        params["filter.phase"] = "|".join(phase)

    if nct_ids:
        params["filter.ids"] = "|".join(nct_ids)

    # Pagination
    if page_token:
        params["pageToken"] = page_token

    # Sorting
    if sort:
        params["sort"] = sort

    return params


def get_studies_url() -> str:
    """Returns the base URL for the studies search endpoint."""
    return CT_API_STUDIES_ENDPOINT


def get_single_study_url(nct_id: str) -> str:
    """Returns the URL to fetch a single study by NCT ID."""
    return f"{CT_API_STUDIES_ENDPOINT}/{nct_id}"