import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Gemini Function Declarations ───────────────────────────────────────
# Each dict follows Gemini's FunctionDeclaration schema:
#   name, description, parameters (JSON Schema)
#
# These tell Gemini WHAT it can call. The ToolRegistry class below
# maps these names to actual Python functions.

TOOL_DECLARATIONS = [
    {
        "name": "medical_term_mapper",
        "description": (
            "Translates everyday language for medical conditions into "
            "standardized medical terminology used by ClinicalTrials.gov. "
            "Call this FIRST before searching. "
            "Example: 'heart attack' → 'myocardial infarction'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The condition or disease term in the user's own words",
                },
            },
            "required": ["term"],
        },
    },
    {
        "name": "trial_searcher",
        "description": (
            "Searches ClinicalTrials.gov for clinical trials matching the "
            "given condition. Use the mapped medical term from "
            "medical_term_mapper when available. Returns a list of trials "
            "with NCT IDs, titles, status, and eligibility info."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "Medical condition to search for (use mapped term if available)",
                },
                "location": {
                    "type": "string",
                    "description": "City, state, or zip code to search near. Optional.",
                },
                "status": {
                    "type": "string",
                    "description": "Trial recruitment status filter",
                    "enum": ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ANY"],
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of trials to return (default 20, max 100)",
                },
            },
            "required": ["condition"],
        },
    },
    {
        "name": "geo_matcher",
        "description": (
            "Filters and ranks a list of trials by geographic proximity "
            "to the user's location. Call this AFTER trial_searcher when "
            "the user has specified a location. Returns trials sorted by "
            "distance with distance info attached."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trial_nct_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of NCT IDs from trial_searcher results to filter",
                },
                "user_location": {
                    "type": "string",
                    "description": "User's location (city, state or zip code)",
                },
                "max_distance_miles": {
                    "type": "number",
                    "description": "Maximum distance in miles (default 50)",
                },
            },
            "required": ["trial_nct_ids", "user_location"],
        },
    },
    {
        "name": "eligibility_parser",
        "description": (
            "Checks whether a user meets a trial's eligibility criteria "
            "based on age, gender, and medical conditions. Call this when "
            "the user has provided personal details. Returns eligible/not "
            "eligible with specific reasons."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trial_nct_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "NCT IDs of trials to check eligibility for",
                },
                "user_age": {
                    "type": "integer",
                    "description": "User's age in years",
                },
                "user_gender": {
                    "type": "string",
                    "description": "User's gender (male/female)",
                },
                "user_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Medical conditions the user has reported",
                },
            },
            "required": ["trial_nct_ids"],
        },
    },
    {
        "name": "plain_language_translator",
        "description": (
            "Translates complex medical text into simple, everyday English. "
            "Call this BEFORE presenting trial descriptions or eligibility "
            "criteria to the user. Makes results accessible to non-medical "
            "audiences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The medical text to simplify",
                },
                "context": {
                    "type": "string",
                    "description": "What this text is (e.g., 'trial description', 'eligibility criteria')",
                    "enum": ["trial_description", "eligibility_criteria", "condition_info", "general"],
                },
            },
            "required": ["text"],
        },
    },
]


# ── Tool Registry Class ───────────────────────────────────────────────

class ToolRegistry:
    """
    Maps tool names to their implementations and provides
    the declarations list for Gemini's function calling setup.

    Usage:
        registry = ToolRegistry()
        registry.register("trial_searcher", my_search_function)
        result = registry.execute("trial_searcher", {"condition": "diabetes"})
    """

    def __init__(self):
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._declarations = {d["name"]: d for d in TOOL_DECLARATIONS}

    def register(self, name: str, handler: Callable[..., Any]):
        """Register a Python function as the handler for a tool name."""
        if name not in self._declarations:
            logger.warning(
                f"Registering handler for '{name}' which has no declaration. "
                "Gemini won't know about this tool unless you add a declaration."
            )
        self._handlers[name] = handler
        logger.info(f"Registered tool: {name}")

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """
        Execute a tool by name with the given arguments.
        Called by the agent when Gemini returns a function_call.
        """
        if name not in self._handlers:
            raise ValueError(
                f"No handler registered for tool '{name}'. "
                f"Available: {list(self._handlers.keys())}"
            )
        logger.info(f"Executing tool: {name} with args: {list(args.keys())}")
        try:
            return self._handlers[name](**args)
        except TypeError as e:
            logger.error(f"Argument mismatch calling '{name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
            raise

    def get_declarations(self) -> list[dict]:
        """Return tool declarations in the format Gemini expects."""
        return TOOL_DECLARATIONS

    def get_registered_names(self) -> list[str]:
        """Return names of tools that have handlers registered."""
        return list(self._handlers.keys())

    def is_registered(self, name: str) -> bool:
        return name in self._handlers