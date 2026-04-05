"""
Clinical Trial Finder — Streamlit Web App
A user-friendly interface for searching clinical trials.
Run with: streamlit run app.py
"""

import sys
import os
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import GROQ_API_KEY, GROQ_MODEL
from src.api.client import ClinicalTrialsClient
from src.tools.trial_searcher import TrialSearcher, SearchParams
from src.tools.medical_term_mapper import MedicalTermMapper
from src.tools.eligibility_parser import EligibilityParser, UserProfile
from src.tools.plain_language import PlainLanguageTranslator, ContentType
from src.tools.geo_matcher import GeoMatcher
from src.agent.tool_registry import ToolRegistry
from src.agent.agent import ClinicalTrialAgent


# ── Agent Setup (cached so it doesn't rebuild on every rerun) ──────────

@st.cache_resource
def build_agent():
    """Build and cache the agent so it persists across reruns."""
    api_client = ClinicalTrialsClient()
    term_mapper = MedicalTermMapper()
    eligibility_parser = EligibilityParser()
    plain_language = PlainLanguageTranslator()
    geo_matcher = GeoMatcher()

    _cached_trials: list = []

    def stub_geo_matcher(**kwargs) -> dict:
        return {"message": "Geo matcher not yet implemented", "results": []}

    searcher = TrialSearcher(
        api_client=api_client,
        term_mapper=lambda term: term_mapper.map_term(term).preferred_term,
    )

    def medical_term_mapper_handler(term: str) -> dict:
        result = term_mapper.map_term(term)
        return {
            "original_term": result.original_term,
            "preferred_term": result.preferred_term,
            "confidence": result.confidence,
            "match_type": result.match_type,
            "alternatives": result.alternatives,
        }

    def trial_searcher_handler(
        condition: str,
        location: str = None,
        status: str = "RECRUITING",
        max_results: int = 5,
    ) -> dict:
        max_results = min(max_results, 3)
        if status and status.upper() == "ANY":
            status = "RECRUITING"
        params = SearchParams(
            condition=condition,
            location=location,
            status=status,
            max_results=max_results,
        )
        result = searcher.search(params)
        _cached_trials.clear()
        _cached_trials.extend(result.trials)
        return {
            "total_found": result.total_found,
            "query_used": result.query_used,
            "filters_applied": result.filters_applied,
            "errors": result.errors,
            "trials": [_trial_to_dict(t) for t in result.trials],
        }

    def eligibility_parser_handler(
        trial_nct_ids: list[str],
        user_age: int = None,
        user_gender: str = None,
        user_conditions: list[str] = None,
    ) -> dict:
        profile = UserProfile(
            age=user_age,
            gender=user_gender,
            conditions=user_conditions or [],
        )
        results = []
        for nct_id in trial_nct_ids:
            try:
                trial_data = searcher.client.get_study(nct_id)
                elig_text = ""
                if trial_data.eligibility and trial_data.eligibility.criteria_text:
                    elig_text = trial_data.eligibility.criteria_text
                result = eligibility_parser.check_eligibility(elig_text, profile)
                results.append({
                    "nct_id": nct_id,
                    "eligible": result.eligible,
                    "summary": result.summary,
                    "met_criteria": result.met_criteria,
                    "unmet_criteria": result.unmet_criteria,
                })
            except Exception as e:
                results.append({
                    "nct_id": nct_id,
                    "eligible": "uncertain",
                    "summary": f"Could not check eligibility: {e}",
                })
        return {"results": results}

    def plain_language_handler(text: str, context: str = "general") -> str:
        try:
            content_type = ContentType(context) if context in ContentType._value2member_map_ else ContentType.GENERAL
            result = plain_language.translate(text, content_type=content_type)
            return result.plain_text
        except Exception:
            return text

    def geo_matcher_handler(
        trial_nct_ids: list[str],
        user_location: str,
        max_distance_miles: float = 50.0,
    ) -> dict:
        trials_to_match = [t for t in _cached_trials if t.nct_id in trial_nct_ids]
        if not trials_to_match:
            trials_to_match = _cached_trials
        summary = geo_matcher.match(
            trials=trials_to_match,
            user_location=user_location,
            radius_miles=max_distance_miles,
        )
        return {
            "user_location": summary.user_location,
            "radius_miles": summary.radius_miles,
            "total_input": summary.total_trials_input,
            "within_radius": summary.trials_within_radius,
            "errors": summary.errors,
            "results": [
                {
                    "nct_id": r.trial.nct_id,
                    "distance_miles": round(r.distance_miles, 1),
                    "facility": r.facility_name,
                    "city": r.city,
                    "state": r.state,
                }
                for r in summary.results
            ],
        }

    registry = ToolRegistry()
    registry.register("medical_term_mapper", medical_term_mapper_handler)
    registry.register("trial_searcher", trial_searcher_handler)
    registry.register("geo_matcher", geo_matcher_handler)
    registry.register("eligibility_parser", eligibility_parser_handler)
    registry.register("plain_language_translator", plain_language_handler)

    return ClinicalTrialAgent(registry=registry)


def _trial_to_dict(trial) -> dict:
    summary = getattr(trial, "brief_summary", None)
    if summary and len(summary) > 300:
        summary = summary[:300] + "... [truncated]"
    result = {
        "nct_id": getattr(trial, "nct_id", None),
        "brief_title": getattr(trial, "brief_title", None),
        "overall_status": getattr(trial, "overall_status", None),
        "phase": getattr(trial, "phase", None),
        "conditions": getattr(trial, "conditions", None),
        "brief_summary": summary,
        "sponsor": getattr(trial, "sponsor", None),
    }
    locations = getattr(trial, "locations", None)
    if locations:
        result["locations"] = [
            {
                "facility": getattr(loc, "facility", None),
                "city": getattr(loc, "city", None),
                "state": getattr(loc, "state", None),
                "country": getattr(loc, "country", None),
            }
            for loc in locations[:3]
        ]
        if len(locations) > 3:
            result["total_locations"] = len(locations)
    elig = getattr(trial, "eligibility", None)
    if elig:
        criteria = getattr(elig, "criteria_text", None)
        if criteria and len(criteria) > 500:
            criteria = criteria[:500] + "... [truncated]"
        result["eligibility"] = {
            "minimum_age": getattr(elig, "minimum_age", None),
            "maximum_age": getattr(elig, "maximum_age", None),
            "gender": getattr(elig, "gender", None),
            "criteria_text": criteria,
        }
    contacts = getattr(trial, "contacts", None)
    if contacts:
        result["contacts"] = [
            {
                "name": getattr(c, "name", None),
                "phone": getattr(c, "phone", None),
                "email": getattr(c, "email", None),
            }
            for c in contacts
        ]
    return result


# ── Page Config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Trial Finder",
    page_icon="🔬",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────

st.title("🔬 Clinical Trial Finder")
st.caption("Search ClinicalTrials.gov and understand eligibility in plain English.")

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown(
        "This tool helps you find clinical trials on "
        "[ClinicalTrials.gov](https://clinicaltrials.gov) and explains "
        "eligibility criteria in simple language."
    )
    st.divider()
    st.markdown(
        "⚠️ **This is NOT medical advice.** Always consult your "
        "healthcare provider before making decisions about clinical "
        "trial participation."
    )
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        agent = build_agent()
        agent.reset_conversation()
        st.rerun()

# ── Chat Interface ─────────────────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("e.g., Type 2 diabetes trials in Boston"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching for clinical trials..."):
            try:
                agent = build_agent()
                response = agent.run(prompt)
            except Exception as e:
                response = (
                    f"I'm sorry, something went wrong: {e}\n\n"
                    "You can try searching directly at "
                    "[ClinicalTrials.gov](https://clinicaltrials.gov)."
                )
        st.markdown(response)

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})