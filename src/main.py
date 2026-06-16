from pathlib import Path
import argparse
import logging
import subprocess
from time import perf_counter
from typing import Any

from langchain_ollama import ChatOllama, OllamaLLM

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


def download_ollama_model(model_name: str):
    logging.info(f"Prüfe/Lade Ollama-Modell '{model_name}'...")
    try:
        # Führt den CLI-Befehl aus. Ist das Modell bereits da, prüft Ollama nur auf Updates.
        cmd = ["ollama", "pull", model_name]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            logging.log(msg=line.strip(), level=1)

        process.wait()
        if process.returncode == 0:
            logging.info(f"Modell '{model_name}' erfolgreich bereit.")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except Exception as e:
        logging.error(
            f"Fehler beim Laden des Modells '{model_name}': {e.stderr.decode()}"
        )
        raise


def get_llm(provider: str, model: str, ollama_model_parameters: dict):
    # Nicht mehr möglich, da in den Klassen, die llm types verwendet werden...
    # if provider == "vertex":
    #     return ChatVertexAI(model_name="gemini-3-pro", temperature=0)
    if provider == "ollama":
        logging.info("Loading Ollama LLM")
        download_ollama_model(model)
        # Temperatur auf 0.0 für deterministische Antworten setzen
        return ChatOllama(**ollama_model_parameters)
    else:
        raise ValueError(f"Provider {provider} nicht unterstützt.")


def run_experiment():
    try:
        logging.info("Starting Document Automation Benchmark Experiment")
        parser = argparse.ArgumentParser(description="Document Automation Benchmark")
        parser.add_argument(
            "--condition",
            type=str,
            choices=["all", "c1", "c2", "c3", "c4"],
            default="all",
            help="Which condition(s) to run",
        )
        parser.add_argument(
            "--experiment",
            type=str,
            choices=["A", "B"],
            default="A",
            help="Teilexperiment (A oder B)",
        )
        parser.add_argument(
            "--complexity",
            type=str,
            choices=["all", "L1", "L2", "L3"],
            default="all",
            help="Complexity level of documents to test on",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of documents to process",
        )
        parser.add_argument(
            "--provider",
            type=str,
            choices=["vertex", "ollama"],
            default="ollama",
            help="LLM provider to use for conditions C2-C4",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="llama3",
            help="Name of the Ollama model to use",
        )

        args = parser.parse_args()

        evaluator = BenchmarkEvaluator(results_dir=str(PROJECT_ROOT / "results"))
        data_dir_path = str(PROJECT_ROOT / "data" / "processed")
        loader = DataLoader(data_dir=data_dir_path, experiment=args.experiment)

        llm_params = {
            "model": args.model,
            "temperature": 0.0,
            "format": "json",
            "seed": 48,
            "top_k": 5,
            "top_p": 0.1,
        }
        llm = get_llm(args.provider, args.model, llm_params)

        documents = loader.load_docs(complexity=args.complexity, limit=args.limit)

        available_conditions = {
            "c1": RuleBasedCondition(),
            "c2": SinglePromptCondition(llm=llm),
            # "c3": SingleAgentCondition(llm=llm),
            # "c4": MultiAgentCondition(llm=llm),
        }

        conditions_to_run = (
            available_conditions
            if args.condition == "all"
            else {args.condition: available_conditions[args.condition]}
        )

        for doc in documents:
            for condition_id, condition_instance in conditions_to_run.items():
                logging.info(
                    f"Running {condition_id} on Document {doc.id} (Complexity: {doc.complexity})"
                )

                start = perf_counter()
                prediction, meta_data = condition_instance.extract_data(doc)
                duration = perf_counter() - start

                evaluator.evaluate(
                    condition_id=condition_id,
                    complexity_level=doc.complexity,
                    doc_id=doc.id,
                    predicted_data=prediction,
                    ground_truth_data=doc.ground_truth,
                    metadata=meta_data,
                    duration=duration,
                )

        evaluator.save_to_csv(args.experiment, args.complexity)

    except Exception as e:
        logging.error(f"Experiment execution failed: {e}")
        raise e
