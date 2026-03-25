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

from config import GEMINI_API_KEY, GEMINI_MODEL
from src.api.client import ClinicalTrialsClient
from src.tools.trial_searcher import TrialSearcher
from src.tools.medical_term_mapper import MedicalTermMapper
from src.tools.eligibility_parser import EligibilityParser, UserProfile
from src.tools.plain_language import PlainLanguageTranslator, ContentType
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
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found.")
        print("Create a .env file in the project root with:")
        print("  GEMINI_API_KEY=your_key_here")
        sys.exit(1)
    print(f"Model: {GEMINI_MODEL}")


def build_agent() -> ClinicalTrialAgent:
    """Wire up all components and return a ready-to-use agent."""

    # 1. API client
    api_client = ClinicalTrialsClient()

    # 2. Member 2's tools — real implementations
    term_mapper = MedicalTermMapper()
    eligibility_parser = EligibilityParser(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    plain_language = PlainLanguageTranslator(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)

    #    Member 3's tool (stub until delivered):
    def stub_geo_matcher(**kwargs) -> dict:
        """Stub until Member 3 delivers geo_matcher."""
        return {"message": "Geo matcher not yet implemented", "results": []}

    # 3. Trial searcher with real term mapper
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
        max_results: int = 20,
    ) -> dict:
        """Adapter: converts Gemini's args into SearchParams and runs search."""
        from src.tools.trial_searcher import SearchParams

        params = SearchParams(
            condition=condition,
            location=location,
            status=status,
            max_results=max_results,
        )
        result = searcher.search(params)
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
        return result.translated_text

    # 5. Register all tools
    registry = ToolRegistry()
    registry.register("medical_term_mapper", medical_term_mapper_handler)
    registry.register("trial_searcher", trial_searcher_handler)
    registry.register("geo_matcher", stub_geo_matcher)
    registry.register("eligibility_parser", eligibility_parser_handler)
    registry.register("plain_language_translator", plain_language_handler)

    # 5. Build and return the agent
    agent = ClinicalTrialAgent(registry=registry)
    print(f"Agent ready. {len(registry.get_registered_names())} tools registered:")
    for name in registry.get_registered_names():
        print(f"  - {name}")

    return agent


def _trial_to_dict(trial) -> dict:
    """Convert a Trial object to a clean dict for Gemini."""
    result = {
        "nct_id": getattr(trial, "nct_id", None),
        "brief_title": getattr(trial, "brief_title", None),
        "official_title": getattr(trial, "official_title", None),
        "overall_status": getattr(trial, "overall_status", None),
        "phase": getattr(trial, "phase", None),
        "conditions": getattr(trial, "conditions", None),
        "brief_summary": getattr(trial, "brief_summary", None),
        "sponsor": getattr(trial, "sponsor", None),
        "start_date": getattr(trial, "start_date", None),
        "completion_date": getattr(trial, "completion_date", None),
    }

    # Locations
    locations = getattr(trial, "locations", None)
    if locations:
        result["locations"] = [
            {
                "facility": getattr(loc, "facility", None),
                "city": getattr(loc, "city", None),
                "state": getattr(loc, "state", None),
                "country": getattr(loc, "country", None),
            }
            for loc in locations
        ]

    # Eligibility
    elig = getattr(trial, "eligibility", None)
    if elig:
        result["eligibility"] = {
            "minimum_age": getattr(elig, "minimum_age", None),
            "maximum_age": getattr(elig, "maximum_age", None),
            "gender": getattr(elig, "gender", None),
            "criteria_text": getattr(elig, "criteria_text", None),
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