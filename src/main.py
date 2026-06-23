from pathlib import Path
import argparse
import logging
import subprocess
from time import perf_counter

import requests

from langchain_ollama import ChatOllama

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


def is_ollama_server_running(
    host: str = "http://localhost:11434", timeout: float = 2.0
) -> bool:
    try:
        resp = requests.get(f"{host}/api/version", timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def download_ollama_model(model_name: str):
    logging.info(f"Prüfe/Lade Ollama-Modell '{model_name}'...")
    try:
        if is_ollama_server_running():
            resp = requests.post(
                "http://localhost:11434/api/pull",
                json={"model": model_name, "stream": False},
                timeout=None,
            )
            resp.raise_for_status()
        else:
            cmd = ["ollama", "pull", model_name]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                logging.info(line.strip())
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        logging.info(f"Modell '{model_name}' erfolgreich bereit.")
    except Exception as e:
        logging.error(f"Fehler beim Laden des Modells '{model_name}': {e}")
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
            nargs="+",
            choices=["all", "C1", "C2", "C3", "C4"],
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
            nargs="+",
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
            nargs="+",
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

        base_llm_params = {
            "model": args.model,
            "temperature": 0.0,
            "seed": 48,
            "top_k": 5,
            "top_p": 0.1,
            "num_ctx": 16384,
        }
        llm_text = get_llm(args.provider, args.model, base_llm_params)

        json_llm_params = base_llm_params.copy()
        json_llm_params["format"] = "json"
        llm_json = get_llm(args.provider, args.model, json_llm_params)

        documents = loader.load_docs(complexity=args.complexity, limit=args.limit)

        available_conditions = {
            "C1": RuleBasedCondition(),
            "C2": SinglePromptCondition(llm=llm_json),
            "C3": SingleAgentCondition(llm=llm_text),
            "C4": MultiAgentCondition(llm_text=llm_text, llm_json=llm_json),
        }

        if "all" in args.condition:
            conditions_to_run = available_conditions
        else:
            conditions_to_run = {
                key: available_conditions[key]
                for key in args.condition
                if key in available_conditions
            }

        for idx, doc in enumerate(documents):
            for condition_id, condition_instance in conditions_to_run.items():
                logging.info(
                    f"Running {condition_id} on Document {doc.id} (Complexity: {doc.complexity}) (Doc: {idx}/{len(documents)})"
                )

                prediction: dict = {}
                meta_data: dict = {}
                error_msg = None

                start = perf_counter()
                try:
                    # error_msg ist None bei Erfolg, ein String bei internen
                    prediction, meta_data, error_msg = condition_instance.extract_data(
                        doc
                    )
                except Exception as e:
                    # Harter System-Absturz den die Condition nicht intern abgefangen hat.
                    error_msg = f"{type(e).__name__}: {e}"
                    logging.error(
                        f"Hard failure in {condition_id} for doc {doc.id}: {e}"
                    )
                finally:
                    duration = perf_counter() - start

                evaluator.evaluate(
                    condition_id=condition_id,
                    complexity_level=doc.complexity,
                    doc_id=doc.id,
                    predicted_data=prediction,
                    ground_truth_data=doc.ground_truth,
                    doc_text=doc.content,
                    metadata=meta_data,
                    duration=duration,
                    model=args.model if condition_id == "C1" "" else args.model,
                    error=error_msg,
                )

        evaluator.save_to_csv(args.experiment, args.complexity)

    except Exception as e:
        logging.error(f"Experiment execution failed: {e}")
        raise e
