from pathlib import Path
import argparse
import logging

from src.evaluation import BenchmarkEvaluator

# Basic Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run the Document Automation Benchmark"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Select the system configuration (Condition C1-C4) to evaluate",
        choices=[
            "all",
            "c1_rule_based",
            "c2_single_prompt",
            "c3_single_agent",
            "c4_multi_agent",
        ],
        default="all",
    )

    parser.add_argument(
        "--complexity",
        type=str,
        help="Select the document complexity level (L1-L3)",
        choices=["all", "l1", "l2", "l3"],
        default="all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of documents to process for testing purposes",
        default=None,
    )

    args = parser.parse_args()

    try:
        evaluator = BenchmarkEvaluator(
            results_dir=str(PROJECT_ROOT / "results"),
        )
        if args.config in ["all", "c1_rule_based"]:
            from src.architectures.c1_rule_based import run_condition_c1

            logging.info("Starting C1 (Regex/RPA) Evaluation...")
            run_condition_c1(
                complexity=args.complexity, limit=args.limit, evaluator=evaluator
            )

        if args.config in ["all", "c2_single_prompt"]:
            from architectures.c2_singe_prompt import run_condition_c2

            logging.info("Starting C2 (Single-Prompt LLM) Evaluation...")
            run_condition_c2(
                complexity=args.complexity, limit=args.limit, evaluator=evaluator
            )

        if args.config in ["all", "c3_single_agent"]:
            from architectures.c3_ai_agent import run_condition_c3

            logging.info("Starting C3 (Single Agent) Evaluation...")
            run_condition_c3(complexity=args.complexity, limit=args.limit)

        if args.config in ["all", "c4_multi_agent"]:
            from architectures.c4_multi_ai_agents import run_condition_c4

            logging.info("Starting C4 (Multi Agent) Evaluation...")
            run_condition_c4(complexity=args.complexity, limit=args.limit)

        evaluator.evaluate(
            condition_id=args.config,
            complexity_level=args.complexity,
            doc_id="doc_001",
            predicted_data={"field1": "value1", "field2": "value2"},
            metadata={"tokens": 150, "duration": 2.5},
        )
        evaluator.save_to_csv(args.config, args.complexity)
    except Exception as e:
        logging.error(f"Experiment execution failed: {e}")
        raise e


if __name__ == "__main__":
    main()
