# src/architectures/c4_multi_ai_agents.py
import json
import logging
import traceback
from typing import Dict, Optional, Tuple, List, TypedDict, Any
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.architectures.base import BaseCondition
from src.data_loader import Document


# Pydantic Schemata für Planner und Validator
class PlannerOutput(BaseModel):
    reasoning: str = Field(description="Detailed analysis of document structure")
    strategy: str = Field(
        description="Step-by-step extraction instructions including which tools to use"
    )


class ValidatorOutput(BaseModel):
    status: str = Field(description="Must be exactly 'PASSED' or 'FAILED'")
    feedback: str = Field(
        description="Detailed feedback on hallucinations or missing values"
    )


# MAS State
class MultiAgentState(TypedDict):
    document_content: str
    target_fields: List[str]
    schema_class: Optional[type]
    planner_reasoning: str
    planner_strategy: str
    raw_extraction: Dict[str, Any]
    intermediate_steps: List[Tuple]
    final_output: Dict[str, Any]
    validation_errors: str
    correction_count: int
    input_tokens: int
    output_tokens: int
    used_tools: List[str]


class MultiAgentCondition(BaseCondition):
    def __init__(self, llm_text: BaseChatModel, llm_json: BaseChatModel):
        self.llm_text = llm_text
        self.llm_json = llm_json
        self.max_retries = 3
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        from langgraph.graph import StateGraph, START, END

        workflow = StateGraph(MultiAgentState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("extractor_with_tools", self._extractor_node)
        workflow.add_node("validator", self._validator_node)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "extractor_with_tools")
        workflow.add_edge("extractor_with_tools", "validator")

        workflow.add_conditional_edges(
            "validator",
            self._should_continue,
            {"continue": "extractor_with_tools", "end": END},
        )

        return workflow.compile()

    def _build_extractor_agent(self):
        """Build agent WITH tool support"""
        return None  # Lazy load below

    def _planner_node(self, state: MultiAgentState) -> Dict:
        """Plan extraction strategy WITHOUT tool calls"""
        self.logger.info("C4: Planning...")

        prompt = [
            SystemMessage(
                content=(
                    f"You are a Planner Agent. Target Fields: {', '.join(state['target_fields'])}\n"
                    "Analyze structure, identify tables, metadata locations.\n"
                    "DO NOT EXTRACT DATA YOURSELF!\n"
                    "Recommend IF and WHERE tools might help (search, locate)."
                )
            ),
            HumanMessage(content=state["document_content"]),
        ]

        planner_llm = self.llm_json.with_structured_output(
            PlannerOutput, include_raw=True
        )
        response = planner_llm.invoke(prompt)

        parsed = response.get("parsed")
        raw = response.get("raw")

        reasoning = parsed.reasoning if parsed else "Analysis failed."
        strategy = parsed.strategy if parsed else "Default extraction."

        tokens_in = (
            raw.usage_metadata.get("input_tokens", 0)
            if raw and hasattr(raw, "usage_metadata")
            else 0
        )
        tokens_out = (
            raw.usage_metadata.get("output_tokens", 0)
            if raw and hasattr(raw, "usage_metadata")
            else 0
        )

        return {
            "planner_reasoning": reasoning,
            "planner_strategy": strategy,
            "input_tokens": state.get("input_tokens", 0) + tokens_in,
            "output_tokens": state.get("output_tokens", 0) + tokens_out,
        }

    def _extractor_node(self, state: MultiAgentState) -> Dict:
        """Extractor with TOOLS available BUT final output must match schema"""
        self.logger.info(
            f"C4: Extracting with tools (Attempt {state.get('correction_count', 0) + 1})"
        )

        from src.architectures.c3_ai_agent import get_document_tools

        ExtractionSchema = state["schema_class"]
        if not ExtractionSchema:
            self.logger.error("C4: No schema available!")
            return {"raw_extraction": {}, "final_output": {}}

        # Get tools for this document
        tools = get_document_tools(state["document_content"])

        # Create agent that can use tools
        react_agent = create_agent(
            model=self.llm_text,
            tools=tools,
        )

        # Build enhanced prompt with strategy + feedback
        strategy = state.get("planner_strategy", "")
        feedback = state.get("validation_errors", "")

        agent_prompt = (
            f"Extraction Task:\n"
            f"Target Fields: {', '.join(state['target_fields'])}\n\n"
            f"Plan/Strategy: {strategy}\n\n"
            f"{'PREVIOUS FEEDBACK:<br>' + feedback + '</FEEDBACK>' if feedback else ''}\n\n"
            f"<Document>\n{state['document_content']}\n</Document>\n\n"
            f"You MAY USE TOOLS to locate information (search_text, extract_table).\n"
            f"After gathering info, format output as JSON matching this exact schema keys only: "
            f"{list(state['target_fields'])}"
        )

        token_tracking = {"in": 0, "out": 0}
        extracted_data = {}
        used_tool_names = []

        try:
            agent_result = react_agent.invoke(agent_prompt)

            # Track tool usage
            if "intermediate_steps" in agent_result:
                for step in agent_result.get("intermediate_steps", []):
                    tool_call = step[0] if step else None
                    if tool_call and hasattr(tool_call, "tool"):
                        used_tool_names.append(tool_call.tool)
                        token_tracking["in"] += getattr(tool_call, "token_usage_in", 0)

            final_msg = agent_result.get("output", "{}")

            # Extract JSON from possible chatter/marker
            if "<extraction>" in final_msg.lower():
                import re

                match = re.search(
                    r"<extraction>(.*?)</extraction>",
                    final_msg,
                    re.DOTALL | re.IGNORECASE,
                )
                if match:
                    final_msg = match.group(1)

            # Validate against Schema
            if isinstance(final_msg, dict):
                extracted_data = final_msg
            else:
                try:
                    # Clean up markdown formatting
                    clean_json_str = final_msg.strip().strip("```json").strip("```")
                    extracted_data = json.loads(clean_json_str)
                except:
                    # Last resort: parse via LLM cleaner pass
                    clean_llm = self.llm_json.with_structured_output(
                        ExtractionSchema, include_raw=True
                    )
                    clean_response = clean_llm.invoke(
                        [HumanMessage(content=f"Convert to JSON:\n{final_msg}")]
                    )
                    parsed = clean_response.get("parsed")
                    if parsed:
                        extracted_data = parsed.model_dump()

        except Exception as e:
            self.logger.error(f"C4 Extractor Error: {e}")
            traceback.print_exc()
            extracted_data = {}

        return {
            "raw_extraction": extracted_data,
            "final_output": extracted_data,
            "intermediate_steps": [
                (step, {}) for step in used_tool_names
            ],  # Store tool usage
            "used_tools": list(set(used_tool_names)),
            "input_tokens": state.get("input_tokens", 0) + token_tracking["in"],
            "output_tokens": state.get("output_tokens", 0) + token_tracking["out"],
        }

    def _validator_node(self, state: MultiAgentState) -> Dict:
        """Validate with original document context"""
        self.logger.info("C4: Validating...")

        data = state.get("raw_extraction", {})
        documents = state["document_content"]

        prompt = [
            SystemMessage(
                content=(
                    "You are a validator. Check extraction quality strictly.\n"
                    "Check: 1) Hallucinations? (Values not in document)\n"
                    "       2) Missing fields? (Required fields null when data exists)\n"
                    "Status must be 'PASSED' or 'FAILED'."
                )
            ),
            HumanMessage(
                content=f"Document:\n{documents[:8000]}...\n\nExtraction:\n{json.dumps(data, indent=2)}"
            ),
        ]

        validator_llm = self.llm_json.with_structured_output(
            ValidatorOutput, include_raw=True
        )
        response = validator_llm.invoke(prompt)

        parsed = response.get("parsed")
        raw = response.get("raw")

        status = parsed.status.upper() if parsed else "FAILED"
        feedback = parsed.feedback if parsed else "Validation error."

        tokens_in = (
            raw.usage_metadata.get("input_tokens", 0)
            if raw and hasattr(raw, "usage_metadata")
            else 0
        )
        tokens_out = (
            raw.usage_metadata.get("output_tokens", 0)
            if raw and hasattr(raw, "usage_metadata")
            else 0
        )

        errors = f"Reviewer: {feedback}" if status != "PASSED" else ""

        return {
            "validation_errors": errors.strip(),
            "correction_count": state.get("correction_count", 0) + 1,
            "input_tokens": state.get("input_tokens", 0) + tokens_in,
            "output_tokens": state.get("output_tokens", 0) + tokens_out,
        }

    def _should_continue(self, state: MultiAgentState) -> str:
        """Decide whether to loop back for correction"""
        has_errors = bool(state.get("validation_errors"))
        retries_left = state.get("correction_count", 0) < self.max_retries

        if has_errors and retries_left:
            self.logger.info(
                f"C4: Retry ({state.get('correction_count', 0)}/{self.max_retries})"
            )
            return "continue"

        return "end"

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        """Main entry point with full tool+schema integration"""

        initial_state = {
            "document_content": document.content,
            "target_fields": document.target_fields,
            "schema_class": document.schema_class,
            "planner_reasoning": "",
            "planner_strategy": "",
            "raw_extraction": {},
            "intermediate_steps": [],
            "final_output": {},
            "validation_errors": "",
            "correction_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "used_tools": [],
        }

        try:
            result_state = self.workflow.invoke(initial_state)

            metadata = {
                "input_tokens": result_state.get("input_tokens", 0),
                "output_tokens": result_state.get("output_tokens", 0),
                "tokens": result_state.get("input_tokens", 0)
                + result_state.get("output_tokens", 0),
                "retries_used": result_state.get("correction_count", 0),
                "tools_used": result_state.get("used_tools", []),
                "intermediate_steps_count": len(
                    result_state.get("intermediate_steps", [])
                ),
            }

            return result_state["raw_extraction"], metadata, None

        except Exception as e:
            self.logger.error(f"C4 Pipeline Error: {e}")
            return (
                {},
                {"input_tokens": 0, "output_tokens": 0, "tokens": 0, "errors": True},
                str(e),
            )
