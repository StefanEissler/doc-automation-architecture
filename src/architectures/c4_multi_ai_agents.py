import json
import logging
import traceback
from typing import Dict, Optional, Tuple, List, TypedDict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.architectures.base import BaseCondition
from src.data_loader import Document


# Pydantic Schemata für Planner und Validator
class PlannerOutput(BaseModel):
    reasoning: str = Field(description="Detailed analysis of document structure")
    strategy: str = Field(description="Step-by-step extraction instructions")


class ValidatorOutput(BaseModel):
    status: str = Field(description="Must be exactly 'PASSED' or 'FAILED'")
    feedback: str = Field(description="Feedback on hallucinations or missing values")


# MAS State
class MultiAgentState(TypedDict):
    """State definition for LangGraph workflow"""

    document_content: str
    target_fields: List[str]
    schema_class: Optional[type]
    planner_reasoning: str
    planner_strategy: str
    raw_extraction: Dict[str, Any]
    final_output: Dict[str, Any]
    validation_errors: str
    correction_count: int
    input_tokens: int
    output_tokens: int


class MultiAgentCondition(BaseCondition):
    def __init__(self, llm_text: BaseChatModel, llm_json: BaseChatModel):
        self.llm_text = llm_text
        self.llm_json = llm_json
        self.max_retries = 3
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Define LangGraph workflow with proper state updates"""
        from langgraph.graph import StateGraph, START, END

        workflow = StateGraph(MultiAgentState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("extractor", self._extractor_node)
        workflow.add_node("validator", self._validator_node)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "extractor")
        workflow.add_edge("extractor", "validator")

        workflow.add_conditional_edges(
            "validator", self._should_continue, {"continue": "extractor", "end": END}
        )

        return workflow.compile()

    def _planner_node(self, state: MultiAgentState) -> Dict:
        """Node 1: Plan extraction strategy"""
        self.logger.info("C4: Planning extraction...")

        prompt = [
            SystemMessage(
                content=(
                    "You are a Senior Data Architect and Planner for broadcast advertising document extraction.\n"
                    "Your ONLY job is to analyze the document structure and write a strategy "
                    "for the Extractor Agent. DO NOT extract any values yourself.\n\n"
                    f"The Extractor must find these fields: {', '.join(state.get('target_fields'))}\n\n"
                    "YOUR TASKS:\n"
                    "1. Identify where in the document each field appears "
                    "   (e.g., 'gross_amount is in the bottom summary table under Gross Total').\n"
                    "2. Flag disambiguation risks:\n"
                    "   - tv_address (station address) vs agency address (different entities)\n"
                    "   - identifier numbers may appear with prefixes like 'Contract #' — strip them\n"
                    "   - IDs appear in parentheses and must be included\n"
                    "3. Note any OCR artifacts or unusual formatting the Extractor should handle.\n\n"
                    "4. Identify the line_items table structure and its fields. If the document contains a table of line items, scan for the table boundries."
                    "Write a numbered, field-by-field strategy the Extractor can follow directly."
                )
            ),
            HumanMessage(
                content=(
                    f"<DOCUMENT>\n{state['document_content']}\n</DOCUMENT>\n\n"
                    f"Write the extraction strategy for: {', '.join(state.get('target_fields'))}"
                )
            ),
        ]

        # Use structured output to guarantee valid JSON response
        planner_llm = self.llm_json.with_structured_output(
            PlannerOutput, include_raw=True
        )

        try:
            response = planner_llm.invoke(prompt)
            parsed = response.get("parsed")
            raw = response.get("raw")

            reasoning = parsed.reasoning if parsed else "Analysis failed."
            strategy = parsed.strategy if parsed else "Use default extraction."

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

        except Exception as e:
            self.logger.error(f"C4 Planner Error: {e}")
            reasoning, strategy = (
                "Planning error occurred.",
                "Default extraction strategy.",
            )
            tokens_in = tokens_out = 0

        return {
            "planner_reasoning": reasoning,
            "planner_strategy": strategy,
            "input_tokens": state.get("input_tokens", 0) + tokens_in,
            "output_tokens": state.get("output_tokens", 0) + tokens_out,
        }

    def _extractor_node(self, state: MultiAgentState) -> Dict:
        """Node 2: Extract data using dynamic schema"""
        self.logger.info(
            f"C4: Extracting (Attempt {state.get('correction_count', 0) + 1})"
        )

        ExtractionSchema = state["schema_class"]

        if not ExtractionSchema:
            self.logger.error("C4: No schema available!")
            return {
                "raw_extraction": {},
                "final_output": {},
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
            }

        schema_json_template = json.dumps(
            ExtractionSchema.model_json_schema(), indent=2
        )

        strategy = state.get("planner_strategy", "")
        feedback = state.get("validation_errors", "")

        feedback_block = (
            (
                f"\n<CORRECTION_REQUIRED>\n"
                f"Your previous extraction had errors. Fix ALL of these:\n{feedback}\n"
                f"</CORRECTION_REQUIRED>\n"
            )
            if feedback
            else ""
        )

        system_prompt = (
            "You are a precise data extraction agent for business documents.\n\n"
            f"Extract ONLY these fields: {', '.join(state['target_fields'])}\n\n"
            "Extract data into this JSON structure:.\n"
            f"{schema_json_template}\n"
            f"PLANNER STRATEGY:\n{strategy}\n\n"
            f"{feedback_block}"
            "GENERAL RULES:\n"
            "- Copy values EXACTLY as they appear in the document — no paraphrasing.\n"
            "- Scalar fields must be plain strings, never dicts or nested objects.\n"
            "- Set a field to null ONLY if the value is genuinely absent from the document.\n"
            "- line_items: list of objects, one entry per broadcast line."
        )

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(f"<DOCUMENT>\n{state['document_content']}\n</DOCUMENT>")
            ),
        ]

        structured_llm = self.llm_text.with_structured_output(
            ExtractionSchema, include_raw=True
        )

        try:
            response = structured_llm.invoke(prompt)
            parsed = response.get("parsed")
            raw = response.get("raw")

            data = parsed.model_dump() if parsed else {}

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

        except Exception as e:
            self.logger.error(f"C4 Extractor Error: {e}")
            data = {}
            tokens_in = tokens_out = 0

        return {
            "raw_extraction": data,
            "final_output": data,
            "input_tokens": state.get("input_tokens", 0) + tokens_in,
            "output_tokens": state.get("output_tokens", 0) + tokens_out,
        }

    def _validator_node(self, state: MultiAgentState) -> Dict:
        """Node 3: Validate extracted data against original document"""
        self.logger.info("C4: Validating extraction quality...")

        data = state.get("raw_extraction", {})

        scalar_fields = [f for f in state.get("target_fields")]
        extracted_subset = {k: v for k, v in data.items() if k in scalar_fields}

        prompt = [
            SystemMessage(
                content=(
                    "You are a strict fact-checker for document extraction.\n\n"
                    "YOUR SCOPE — check ONLY these fields:\n"
                    f"{', '.join(scalar_fields)}\n\n"
                    "CRITICAL RULES:\n"
                    "1. DO NOT mention, require, or check any field not listed above.\n"
                    "   Do not create your own fields and stick to the ones given in this prompt.\n"
                    "2. HALLUCINATION: extracted value cannot be found anywhere in the document text.\n"
                    "3. MISSING: field is in scope AND a clear value exists in the document\n"
                    "   but was extracted as null.\n"
                    "4. NULL IS CORRECT if the value genuinely does not appear in the document.\n"
                    "5. MINOR FORMAT DIFFERENCES are acceptable — flag only factual errors.\n"
                    "   Example: 'KMEG' vs 'KMEG KMEG' is a format issue, not a hallucination.\n"
                    "6. PASSED = no confirmed hallucinations AND no clearly missing in-scope fields.\n"
                    "7. FAILED = at least one confirmed hallucination OR one clearly missing field.\n\n"
                    "Be conservative: only FAIL if you are certain of an error."
                )
            ),
            HumanMessage(
                content=(
                    f"<DOCUMENT>\n{state['document_content']}\n</DOCUMENT>\n\n"
                    f"<EXTRACTED_FIELDS>\n"
                    f"{json.dumps(extracted_subset, indent=2)}\n"
                    f"</EXTRACTED_FIELDS>\n\n"
                    "For each extracted field: verify the value appears in the document. "
                    "Report ONLY errors in the fields listed in YOUR SCOPE."
                )
            ),
        ]

        validator_llm = self.llm_json.with_structured_output(
            ValidatorOutput, include_raw=True
        )

        try:
            response = validator_llm.invoke(prompt)
            parsed = response.get("parsed")
            raw = response.get("raw")

            status = parsed.status.upper() if parsed else "FAILED"
            feedback = parsed.feedback if parsed else "Validation parsing error."

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

        except Exception as e:
            self.logger.error(f"C4 Validator Error: {e}")
            status, feedback = "FAILED", "Critical validation error."
            tokens_in = tokens_out = 0

        errors = f"Reviewer Feedback: {feedback}" if status != "PASSED" else ""

        return {
            "validation_errors": errors.strip(),
            "correction_count": state.get("correction_count", 0) + 1,
            "input_tokens": state.get("input_tokens", 0) + tokens_in,
            "output_tokens": state.get("output_tokens", 0) + tokens_out,
        }

    def _should_continue(self, state: MultiAgentState) -> str:
        """Conditional edge: Decide whether to retry"""
        has_errors = bool(state.get("validation_errors"))
        retries_left = state.get("correction_count", 0) < self.max_retries

        if has_errors and retries_left:
            self.logger.warning(
                f"Validator lehnte ab wegen: {state.get('validation_errors')}"
            )

            self.logger.info(
                f"C4: Correction Loop (Retry {state.get('correction_count', 0) + 1}/{self.max_retries})"
            )
            return "continue"

        self.logger.info(
            f"C4: Validation complete. Final status: {'SUCCESS' if not has_errors else 'FAILED after max retries'}"
        )
        return "end"

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        """Main entry point - invoked by main.py benchmark runner"""

        initial_state = {
            "document_content": document.content,
            "target_fields": document.target_fields,
            "schema_class": document.schema_class,
            "planner_reasoning": "",
            "planner_strategy": "",
            "raw_extraction": {},
            "final_output": {},
            "validation_errors": "",
            "correction_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        try:
            result_state = self.workflow.invoke(initial_state)

            metadata = {
                "input_tokens": result_state.get("input_tokens", 0),
                "output_tokens": result_state.get("output_tokens", 0),
                "tokens": result_state.get("input_tokens", 0)
                + result_state.get("output_tokens", 0),
                "retries_used": result_state.get("correction_count", 0),
                "has_validation_error": bool(result_state.get("validation_errors")),
            }

            return result_state["raw_extraction"], metadata, None

        except Exception as e:
            self.logger.error(f"C4 Pipeline Failure: {e}")
            traceback_str = "\n".join(traceback.format_exc().split("\n")[-5:])
            return (
                {},
                {"input_tokens": 0, "output_tokens": 0, "tokens": 0, "error": True},
                str(e) + "\n" + traceback_str,
            )
