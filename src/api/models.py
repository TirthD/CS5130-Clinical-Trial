"""
Pydantic models for parsing ClinicalTrials.gov API responses.

API response structure:
    study['protocolSection']['identificationModule']['nctId']
    study['protocolSection']['identificationModule']['briefTitle']
    study['protocolSection']['statusModule']['overallStatus']
    study['protocolSection']['designModule']['phases']
    study['protocolSection']['eligibilityModule']
    study['protocolSection']['contactsLocationsModule']['locations']
    study['protocolSection']['armsInterventionsModule']['interventions']
    study['protocolSection']['descriptionModule']['briefSummary']
"""

from typing import Optional
from pydantic import BaseModel


class Location(BaseModel):
    """A single trial site location."""
    facility: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    status: Optional[str] = None


class Contact(BaseModel):
    """Contact information for a trial."""
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class Intervention(BaseModel):
    """An intervention (drug, device, procedure, etc.)."""
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None


class Eligibility(BaseModel):
    """Eligibility criteria for a trial."""
    criteria_text: Optional[str] = None
    gender: Optional[str] = None
    minimum_age: Optional[str] = None
    maximum_age: Optional[str] = None
    healthy_volunteers: Optional[bool] = None


class Trial(BaseModel):
    """
    Cleaned, flat representation of a single clinical trial.
    Parsed from the nested API response structure.
    """
    nct_id: str
    brief_title: Optional[str] = None
    official_title: Optional[str] = None
    overall_status: Optional[str] = None
    phase: Optional[list[str]] = None
    conditions: Optional[list[str]] = None
    interventions: Optional[list[Intervention]] = None
    brief_summary: Optional[str] = None
    eligibility: Optional[Eligibility] = None
    locations: Optional[list[Location]] = None
    contacts: Optional[list[Contact]] = None
    sponsor: Optional[str] = None
    start_date: Optional[str] = None
    completion_date: Optional[str] = None

    @classmethod
    def from_api_response(cls, study: dict) -> "Trial":
        """
        Parse a single study from the raw API JSON response
        into a clean Trial object.
        """
        protocol = study.get("protocolSection", {})

        # Identification
        id_mod = protocol.get("identificationModule", {})
        nct_id = id_mod.get("nctId", "UNKNOWN")
        brief_title = id_mod.get("briefTitle")
        official_title = id_mod.get("officialTitle")

        # Status
        status_mod = protocol.get("statusModule", {})
        overall_status = status_mod.get("overallStatus")
        start_date = status_mod.get("startDateStruct", {}).get("date")
        completion_date = status_mod.get("primaryCompletionDateStruct", {}).get("date")

        # Design (phase)
        design_mod = protocol.get("designModule", {})
        phase = design_mod.get("phases")

        # Conditions
        cond_mod = protocol.get("conditionsModule", {})
        conditions = cond_mod.get("conditions")

        # Description
        desc_mod = protocol.get("descriptionModule", {})
        brief_summary = desc_mod.get("briefSummary")

        # Sponsor
        sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_mod.get("leadSponsor", {})
        sponsor = lead_sponsor.get("name")

        # Interventions
        arms_mod = protocol.get("armsInterventionsModule", {})
        raw_interventions = arms_mod.get("interventions", [])
        interventions = [
            Intervention(
                name=i.get("name"),
                type=i.get("type"),
                description=i.get("description"),
            )
            for i in raw_interventions
        ]

        # Eligibility
        elig_mod = protocol.get("eligibilityModule", {})
        eligibility = Eligibility(
            criteria_text=elig_mod.get("eligibilityCriteria"),
            gender=elig_mod.get("sex"),
            minimum_age=elig_mod.get("minimumAge"),
            maximum_age=elig_mod.get("maximumAge"),
            healthy_volunteers=elig_mod.get("healthyVolunteers"),
        )

        # Locations
        contacts_mod = protocol.get("contactsLocationsModule", {})
        raw_locations = contacts_mod.get("locations", [])
        locations = [
            Location(
                facility=loc.get("facility"),
                city=loc.get("city"),
                state=loc.get("state"),
                country=loc.get("country"),
                zip_code=loc.get("zip"),
                status=loc.get("status"),
            )
            for loc in raw_locations
        ]

        # Contacts
        central_contacts = contacts_mod.get("centralContacts", [])
        contacts = [
            Contact(
                name=c.get("name"),
                phone=c.get("phone"),
                email=c.get("email"),
            )
            for c in central_contacts
        ]

        return cls(
            nct_id=nct_id,
            brief_title=brief_title,
            official_title=official_title,
            overall_status=overall_status,
            phase=phase,
            conditions=conditions,
            interventions=interventions,
            brief_summary=brief_summary,
            eligibility=eligibility,
            locations=locations,
            contacts=contacts,
            sponsor=sponsor,
            start_date=start_date,
            completion_date=completion_date,
        )


class SearchResult(BaseModel):
    """Wrapper for a search response with multiple trials."""
    total_count: int = 0
    trials: list[Trial] = []
    next_page_token: Optional[str] = None