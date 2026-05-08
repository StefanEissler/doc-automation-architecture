from pathlib import Path
import argparse
import logging
from time import perf_counter

from langchain_ollama import OllamaLLM

from src.architectures.c1_rule_based import RuleBasedCondition
from src.architectures.c2_singe_prompt import SinglePromptCondition
from src.architectures.c3_ai_agent import SingleAgentCondition
from src.architectures.c4_multi_ai_agents import MultiAgentCondition
from src.data_loader import DataLoader
from src.evaluation import BenchmarkEvaluator

# Basic Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_llm(provider: str):
    # if provider == "vertex":
    #     return ChatVertexAI(model_name="gemini-3-pro", temperature=0)
    if provider == "ollama":
        logging.info("Loading Ollama LLM")
        return OllamaLLM(model="llama3", temperature=0.2)
    else:
        raise ValueError(f"Provider {provider} nicht unterstützt.")


def run_experiment():
    try:
        logging.info("Starting Document Automation Benchmark Experiment")
        parser = argparse.ArgumentParser(description="Document Automation Benchmark")
        parser.add_argument(
            "--config", type=str, choices=["all", "c1", "c2", "c3", "c4"], default="all"
        )
        parser.add_argument(
            "--complexity", type=str, choices=["all", "L1", "L2", "L3"], default="all"
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--provider", type=str, choices=["vertex", "ollama"], default="ollama"
        )

        args = parser.parse_args()

        evaluator = BenchmarkEvaluator(results_dir=str(PROJECT_ROOT / "results"))
        loader = DataLoader(data_dir=str(PROJECT_ROOT / "data" / "corpus"))
        llm = get_llm(args.provider)

        documents = loader.load_docs(complexity=args.complexity, limit=args.limit)

        available_conditions = {
            "c1": RuleBasedCondition(),
            "c2": SinglePromptCondition(llm=llm),
            "c3": SingleAgentCondition(llm=llm),
            "c4": MultiAgentCondition(llm=llm),
        }

        conditions_to_run = (
            available_conditions
            if args.config == "all"
            else {args.config: available_conditions[args.config]}
        )

        for doc in documents:
            for condition_id, condition_instance in conditions_to_run.items():
                logging.info(
                    f"Running {condition_id} on Document {doc.id} (Complexity: {doc.complexity})"
                )

                start = perf_counter()
                prediction, tokens = condition_instance.extract_data(doc)
                duration = perf_counter() - start

                evaluator.evaluate(
                    condition_id=condition_id,
                    complexity_level=doc.complexity,
                    doc_id=doc.id,
                    predicted_data=prediction,
                    metadata={"tokens": tokens, "duration": duration},
                )

        evaluator.save_to_csv(args.config, args.complexity)

    except Exception as e:
        logging.error(f"Experiment execution failed: {e}")
        raise e


if __name__ == "__main__":
    run_experiment()
