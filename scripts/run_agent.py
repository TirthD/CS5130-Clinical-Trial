"""
Run the Clinical Trial Finder agent interactively from the terminal.

Usage:
    python scripts/run_agent.py
    python scripts/run_agent.py --query "diabetes trials in Boston"
"""

import sys
import os
import argparse
import logging

# Add project root to path so imports work when running from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import GROQ_API_KEY, GROQ_MODEL
from src.api.client import ClinicalTrialsClient
from src.tools.trial_searcher import TrialSearcher
from src.tools.medical_term_mapper import MedicalTermMapper
from src.tools.eligibility_parser import EligibilityParser, UserProfile
from src.tools.plain_language import PlainLanguageTranslator, ContentType
from src.tools.geo_matcher import GeoMatcher
from src.agent.tool_registry import ToolRegistry
from src.agent.agent import ClinicalTrialAgent

logger = logging.getLogger(__name__)


# ── Setup ──────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def validate_environment():
    """Check that required config is present."""
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found.")
        print("Create a .env file in the project root with:")
        print("  GROQ_API_KEY=your_key_here")
        sys.exit(1)
    print(f"Model: {GROQ_MODEL}")


def build_agent() -> ClinicalTrialAgent:
    """Wire up all components and return a ready-to-use agent."""

    # 1. API client
    api_client = ClinicalTrialsClient()

    # 2. Member 2's tools — real implementations
    term_mapper = MedicalTermMapper()
    eligibility_parser = EligibilityParser()
    plain_language = PlainLanguageTranslator()

    # 3. Member 3's tool — real implementation
    geo_matcher = GeoMatcher()

    # Cache to store Trial objects between tool calls
    _cached_trials: list = []

    # 4. Trial searcher with real term mapper
    searcher = TrialSearcher(
        api_client=api_client,
        term_mapper=lambda term: term_mapper.map_term(term).preferred_term,
    )

    # 4. Tool handler adapters
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
        """Adapter: converts Groq's args into SearchParams and runs search."""
        from src.tools.trial_searcher import SearchParams

        # Cap results to keep token usage within Groq free tier limits
        max_results = min(max_results, 3)

        # Handle "ANY" status — pass None to skip status filter
        if status and status.upper() == "ANY":
            status = "RECRUITING"

        params = SearchParams(
            condition=condition,
            location=location,
            status=status,
            max_results=max_results,
        )
        result = searcher.search(params)
        # Cache trials for geo_matcher to use
        _cached_trials.clear()
        _cached_trials.extend(result.trials)
        return {
            "total_found": result.total_found,
            "query_used": result.query_used,
            "filters_applied": result.filters_applied,
            "errors": result.errors,
            "trials": [_trial_to_dict(t) for t in result.trials],
        }

    def geo_matcher_handler(
        trial_nct_ids: list[str],
        user_location: str,
        max_distance_miles: float = 50.0,
    ) -> dict:
        """Filter cached trials by proximity to user location."""
        # Filter cached trials to only those requested
        trials_to_match = [
            t for t in _cached_trials
            if t.nct_id in trial_nct_ids
        ]
        if not trials_to_match:
            # If no cached trials match, use all cached trials
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
                # Fetch trial eligibility text from cached search results
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
        content_type = ContentType(context) if context in ContentType._value2member_map_ else ContentType.GENERAL
        result = plain_language.translate(text, content_type=content_type)
        return result.plain_text

    # 5. Register all tools
    registry = ToolRegistry()
    registry.register("medical_term_mapper", medical_term_mapper_handler)
    registry.register("trial_searcher", trial_searcher_handler)
    registry.register("geo_matcher", geo_matcher_handler)
    registry.register("eligibility_parser", eligibility_parser_handler)
    registry.register("plain_language_translator", plain_language_handler)

    # 5. Build and return the agent
    agent = ClinicalTrialAgent(registry=registry)
    print(f"Agent ready. {len(registry.get_registered_names())} tools registered:")
    for name in registry.get_registered_names():
        print(f"  - {name}")

    return agent


def _trial_to_dict(trial) -> dict:
    """Convert a Trial object to a clean, token-efficient dict for Groq."""
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

    # Locations — limit to 3 nearest/most relevant
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

    # Eligibility
    elig = getattr(trial, "eligibility", None)
    if elig:
        criteria = getattr(elig, "criteria_text", None)
        # Truncate criteria text to keep token count manageable
        if criteria and len(criteria) > 500:
            criteria = criteria[:500] + "... [truncated]"
        result["eligibility"] = {
            "minimum_age": getattr(elig, "minimum_age", None),
            "maximum_age": getattr(elig, "maximum_age", None),
            "gender": getattr(elig, "gender", None),
            "criteria_text": criteria,
        }

    # Contacts
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


# ── Interactive Loop ───────────────────────────────────────────────────

def run_interactive(agent: ClinicalTrialAgent):
    """Run the agent in an interactive terminal loop."""
    print("\n" + "=" * 60)
    print("  Clinical Trial Finder")
    print("  Type your question, or 'quit' to exit.")
    print("  Type 'reset' to clear conversation history.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "reset":
            agent.reset_conversation()
            print("Conversation reset.\n")
            continue

        print("\nSearching...\n")

        try:
            response = agent.run(query)
            print(f"Agent: {response}\n")
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            print(f"Error: Something went wrong. Please try again.\n")


def run_single_query(agent: ClinicalTrialAgent, query: str):
    """Run a single query and print the result."""
    print(f"\nQuery: {query}\n")
    print("Searching...\n")

    try:
        response = agent.run(query)
        print(f"Agent: {response}\n")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trial Finder Agent"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query instead of interactive mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    validate_environment()

    agent = build_agent()

    if args.query:
        run_single_query(agent, args.query)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()