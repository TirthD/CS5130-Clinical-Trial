"""
Clinical Trial Finder Agent
Main orchestrator that uses Google Gemini to understand user queries,
call tools in sequence, and assemble safe, cited responses.
"""

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_MODEL, GEMINI_API_KEY
from src.agent.prompts import (
    SYSTEM_PROMPT,
    DISCLAIMER,
    build_no_results_response,
    build_tool_error_message,
)
from src.agent.tool_registry import ToolRegistry, TOOL_DECLARATIONS
from src.agent.safety import (
    SafetyGuard,
    is_medical_advice_request,
    get_medical_advice_redirect,
)

logger = logging.getLogger(__name__)

# Maximum rounds of tool calls before we force a final answer
MAX_TOOL_ROUNDS = 10


class ClinicalTrialAgent:
    """
    The main agent class. Receives user queries, orchestrates tool
    calls via Gemini, and returns safe, formatted responses.

    Usage:
        agent = ClinicalTrialAgent(tool_registry)
        response = agent.run("I'm 45 with diabetes in Boston")
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._configure_gemini()
        self._nct_ids_this_query: set[str] = set()
        self._conversation_history: list[types.Content] = []

    def _configure_gemini(self):
        """Set up the Gemini client and model config."""
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = GEMINI_MODEL

        # Build tool declarations in google.genai format
        self.tools = self._build_gemini_tools()

        logger.info(f"Gemini model configured: {self.model_name}")

    def _build_gemini_tools(self) -> list[types.Tool]:
        """Convert our tool declarations into google.genai Tool format."""
        declarations = self.registry.get_declarations()
        function_declarations = []
        for decl in declarations:
            fd = types.FunctionDeclaration(
                name=decl["name"],
                description=decl["description"],
                parameters=decl["parameters"],
            )
            function_declarations.append(fd)
        return [types.Tool(function_declarations=function_declarations)]

    # ── Main Entry Point ───────────────────────────────────────────

    def run(self, user_query: str) -> str:
        """
        Process a user query end-to-end.

        1. Safety pre-check on input
        2. Send to Gemini
        3. Handle tool call loop
        4. Safety post-check on output
        5. Return final response
        """
        logger.info(f"Processing query: {user_query[:100]}...")

        # Step 1: Early safety check — is the user asking for medical advice?
        if is_medical_advice_request(user_query):
            logger.info("Medical advice request detected — redirecting")
            return get_medical_advice_redirect()

        # Reset per-query state
        self._nct_ids_this_query = set()

        # Step 2: Build messages and send to Gemini
        try:
            # Build the contents list: history + new user message
            contents = list(self._conversation_history)
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query)],
                )
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=self.tools,
                ),
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return self._error_response("connecting to the AI model")

        # Step 3: Tool call loop — Gemini may request multiple tools
        final_text = self._handle_tool_loop(contents, response)

        # Step 4: Safety post-check on final response
        final_text = self._apply_safety_checks(final_text)

        # Step 5: Update conversation history for multi-turn
        self._update_history(user_query, final_text)

        return final_text

    # ── Tool Call Loop ─────────────────────────────────────────────

    def _handle_tool_loop(self, contents: list, response) -> str:
        """
        Process Gemini's response, executing tool calls iteratively
        until Gemini returns a final text response.
        """
        rounds = 0

        while rounds < MAX_TOOL_ROUNDS:
            # Check if Gemini wants to call a tool
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # No more tool calls — Gemini is done, extract final text
                return self._extract_text(response)

            # Execute each tool call and collect results
            function_responses = []
            for fc in tool_calls:
                result = self._execute_tool_call(fc)
                function_responses.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response=result,
                    )
                )

            # Append the model's response (with function calls) to contents
            contents.append(response.candidates[0].content)

            # Append tool results as a new content
            contents.append(
                types.Content(
                    role="user",
                    parts=function_responses,
                )
            )

            # Send updated contents back to Gemini
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=self.tools,
                    ),
                )
            except Exception as e:
                logger.error(f"Gemini error after tool results: {e}", exc_info=True)
                return self._error_response("processing tool results")

            rounds += 1

        logger.warning(f"Hit max tool rounds ({MAX_TOOL_ROUNDS})")
        return self._extract_text(response)

    def _extract_tool_calls(self, response) -> list:
        """Extract function calls from Gemini's response."""
        calls = []
        try:
            parts = response.candidates[0].content.parts
            if not parts:
                return calls
            for part in parts:
                if part.function_call:
                    calls.append(part.function_call)
        except (IndexError, AttributeError) as e:
            logger.debug(f"No tool calls found in response: {e}")
        return calls

    def _extract_text(self, response) -> str:
        """Extract text content from Gemini's response."""
        try:
            # Try the .text shortcut first
            if response.text:
                return response.text
        except (ValueError, AttributeError):
            pass

        try:
            parts = response.candidates[0].content.parts
            if not parts:
                return ""
            text_parts = []
            for part in parts:
                if part.text:
                    text_parts.append(part.text)
            return "\n".join(text_parts) if text_parts else ""
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to extract text from response: {e}")
            return ""

    def _execute_tool_call(self, function_call) -> dict[str, Any]:
        """
        Execute a single tool call and return the result.
        Tracks NCT IDs for hallucination detection.
        """
        name = function_call.name
        args = dict(function_call.args) if function_call.args else {}

        logger.info(f"Tool call: {name}({list(args.keys())})")

        try:
            result = self.registry.execute(name, args)

            # Track NCT IDs from search results for safety checks
            self._track_nct_ids(name, result)

            # Serialize result for Gemini
            return {"success": True, "data": self._serialize_result(result)}

        except ValueError as e:
            logger.error(f"Tool '{name}' not found: {e}")
            return {"success": False, "error": f"Tool not available: {name}"}

        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": build_tool_error_message(f"running {name}"),
            }

    def _track_nct_ids(self, tool_name: str, result: Any):
        """
        Extract NCT IDs from tool results so we can verify
        the agent doesn't hallucinate IDs in its final response.
        """
        if tool_name != "trial_searcher":
            return

        try:
            # Handle SearchResult from trial_searcher
            if hasattr(result, "trials"):
                for trial in result.trials:
                    nct_id = getattr(trial, "nct_id", None)
                    if nct_id:
                        self._nct_ids_this_query.add(nct_id)
            # Handle raw list of trials
            elif isinstance(result, list):
                for item in result:
                    nct_id = (
                        getattr(item, "nct_id", None)
                        or (item.get("nct_id") if isinstance(item, dict) else None)
                    )
                    if nct_id:
                        self._nct_ids_this_query.add(nct_id)

            logger.debug(f"Tracking {len(self._nct_ids_this_query)} NCT IDs")
        except Exception as e:
            logger.warning(f"Could not track NCT IDs: {e}")

    @staticmethod
    def _serialize_result(result: Any) -> Any:
        """Convert tool result to JSON-serializable format for Gemini."""
        # Pydantic model
        if hasattr(result, "model_dump"):
            return result.model_dump()

        # Dataclass with a to_dict method
        if hasattr(result, "to_dict"):
            return result.to_dict()

        # Dataclass — convert fields manually
        if hasattr(result, "__dataclass_fields__"):
            data = {}
            for k in result.__dataclass_fields__:
                v = getattr(result, k)
                if isinstance(v, list):
                    data[k] = [
                        ClinicalTrialAgent._serialize_result(item) for item in v
                    ]
                else:
                    data[k] = v
            return data

        # List of objects
        if isinstance(result, list):
            return [ClinicalTrialAgent._serialize_result(item) for item in result]

        # Already serializable (str, int, dict, etc.)
        return result

    # ── Safety ─────────────────────────────────────────────────────

    def _apply_safety_checks(self, response: str) -> str:
        """Run post-generation safety checks and return cleaned response."""
        if not response.strip():
            return DISCLAIMER

        guard = SafetyGuard(known_nct_ids=self._nct_ids_this_query)
        result = guard.check(response)

        if result.issues:
            logger.warning(
                f"Safety check flagged {len(result.issues)} issue(s):"
            )
            for issue in result.issues:
                logger.warning(f"  - {issue}")

        return result.modified_response or response

    # ── Conversation History ───────────────────────────────────────

    def _update_history(self, user_query: str, agent_response: str):
        """
        Maintain conversation history for multi-turn interactions.
        Keeps last N exchanges to avoid context overflow.
        """
        self._conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_query)],
            )
        )
        self._conversation_history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=agent_response)],
            )
        )

        # Keep last 10 exchanges (20 messages) to stay within context limits
        max_messages = 20
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]

    # ── Error Handling ─────────────────────────────────────────────

    def _error_response(self, action: str) -> str:
        """Generate a user-friendly error message."""
        return (
            f"I'm sorry, I ran into a problem while {action}. "
            "Please try again in a moment. If the issue persists, "
            "you can search directly at https://clinicaltrials.gov.\n\n"
            + DISCLAIMER
        )

    # ── Convenience ────────────────────────────────────────────────

    def reset_conversation(self):
        """Clear conversation history for a fresh start."""
        self._conversation_history = []
        self._nct_ids_this_query = set()
        logger.info("Conversation history reset")