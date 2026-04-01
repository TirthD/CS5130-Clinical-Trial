"""
Clinical Trial Finder Agent
Main orchestrator that uses Groq (Llama 3) to understand user queries,
call tools in sequence, and assemble safe, cited responses.
"""

import json
import logging
from typing import Any

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
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
    calls via Groq, and returns safe, formatted responses.

    Usage:
        agent = ClinicalTrialAgent(tool_registry)
        response = agent.run("I'm 45 with diabetes in Boston")
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
        self._nct_ids_this_query: set[str] = set()
        self._conversation_history: list[dict] = []

        # Build tools in OpenAI-compatible format
        self.tools = self._build_tools()
        logger.info(f"Groq model configured: {self.model}")

    def _build_tools(self) -> list[dict]:
        """Convert our tool declarations into OpenAI function calling format."""
        declarations = self.registry.get_declarations()
        return [
            {
                "type": "function",
                "function": {
                    "name": decl["name"],
                    "description": decl["description"],
                    "parameters": decl["parameters"],
                },
            }
            for decl in declarations
        ]

    # ── Main Entry Point ───────────────────────────────────────────

    def run(self, user_query: str) -> str:
        """
        Process a user query end-to-end.

        1. Safety pre-check on input
        2. Send to Groq
        3. Handle tool call loop
        4. Safety post-check on output
        5. Return final response
        """
        logger.info(f"Processing query: {user_query[:100]}...")

        # Step 1: Early safety check
        if is_medical_advice_request(user_query):
            logger.info("Medical advice request detected — redirecting")
            return get_medical_advice_redirect()

        # Reset per-query state
        self._nct_ids_this_query = set()

        # Step 2: Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_query})

        # Step 3: Send to Groq and handle tool loop
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
                tools=self.tools, # type: ignore
                tool_choice="auto",
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}", exc_info=True)
            return self._error_response("connecting to the AI model")

        final_text = self._handle_tool_loop(messages, response)

        # Step 4: Safety post-check
        final_text = self._apply_safety_checks(final_text)

        # Step 5: Update conversation history
        self._update_history(user_query, final_text)

        return final_text

    # ── Tool Call Loop ─────────────────────────────────────────────

    def _handle_tool_loop(self, messages: list[dict], response) -> str:
        """
        Process Groq's response, executing tool calls iteratively
        until Groq returns a final text response.
        """
        rounds = 0

        while rounds < MAX_TOOL_ROUNDS:
            message = response.choices[0].message

            # If no tool calls, we're done
            if not message.tool_calls:
                return message.content or ""

            # Append the assistant's message (with tool calls) to messages
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute each tool call and append results
            for tool_call in message.tool_calls:
                result = self._execute_tool_call(tool_call)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            # Send tool results back to Groq
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,# type: ignore
                    tools=self.tools,# type: ignore
                    tool_choice="auto",
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error(f"Groq error after tool results: {e}", exc_info=True)
                return self._error_response("processing tool results")

            rounds += 1

        logger.warning(f"Hit max tool rounds ({MAX_TOOL_ROUNDS})")
        return response.choices[0].message.content or ""

    def _execute_tool_call(self, tool_call) -> dict[str, Any]:
        """Execute a single tool call and return the result."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}

        logger.info(f"Tool call: {name}({list(args.keys())})")

        try:
            result = self.registry.execute(name, args)

            # Track NCT IDs for hallucination detection
            self._track_nct_ids(name, result)

            # Serialize result
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
        """Extract NCT IDs from tool results for hallucination detection."""
        if tool_name != "trial_searcher":
            return

        try:
            if hasattr(result, "trials"):
                for trial in result.trials:
                    nct_id = getattr(trial, "nct_id", None)
                    if nct_id:
                        self._nct_ids_this_query.add(nct_id)
            elif isinstance(result, dict) and "trials" in result:
                for trial in result["trials"]:
                    nct_id = trial.get("nct_id") if isinstance(trial, dict) else None
                    if nct_id:
                        self._nct_ids_this_query.add(nct_id)
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
        """Convert tool result to JSON-serializable format."""
        # Pydantic model
        if hasattr(result, "model_dump"):
            return result.model_dump()

        # Dataclass with to_dict
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

        # Already serializable
        return result

    # ── Safety ─────────────────────────────────────────────────────

    def _apply_safety_checks(self, response: str) -> str:
        """Run post-generation safety checks and return cleaned response."""
        if not response.strip():
            return DISCLAIMER

        guard = SafetyGuard(known_nct_ids=self._nct_ids_this_query)
        result = guard.check(response)

        if result.issues:
            logger.warning(f"Safety check flagged {len(result.issues)} issue(s):")
            for issue in result.issues:
                logger.warning(f"  - {issue}")

        return result.modified_response or response

    # ── Conversation History ───────────────────────────────────────

    def _update_history(self, user_query: str, agent_response: str):
        """Maintain conversation history for multi-turn interactions."""
        self._conversation_history.append(
            {"role": "user", "content": user_query}
        )
        self._conversation_history.append(
            {"role": "assistant", "content": agent_response}
        )

        # Keep last 10 exchanges (20 messages)
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