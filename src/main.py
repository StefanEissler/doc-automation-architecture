from datetime import datetime
import os
from pathlib import Path
import argparse
import logging
import subprocess
import time
from time import perf_counter

from langchain_google_genai import ChatGoogleGenerativeAI
import requests

from langchain_ollama import ChatOllama

from src.architectures.c1_rule_based import RuleBasedCondition
from src.architectures.c2_single_prompt import SinglePromptCondition
from src.architectures.c3_ai_agent import SingleAgentCondition
from src.architectures.c4_multi_ai_agents import MultiAgentCondition
from src.data_loader import DataLoader
from src.evaluation import BenchmarkEvaluator

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(level_name: str, experiment: str, model: str, provider: str):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    level = getattr(logging, level_name.upper(), logging.INFO)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # (z.B. logs/benchmark_ExpA_google_gemini-3.5-flash_20260707_143000.log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model.replace("/", "-").replace(":", "-")
    log_filename = (
        log_dir
        / f"benchmark_experiment_{experiment}_{provider}_{safe_model_name}_{timestamp}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Log-File will be saved to: {log_filename}")


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


def get_llm(provider: str, model: str, model_parameters: dict):

    if provider == "google":
        logging.info(f"Loading Google Gemini API LLM: {model}")
        if not os.environ.get("GEMINI_API_KEY"):  # noqa: F821
            raise ValueError("GEMINI_API_KEY environment variable is missing!")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=model_parameters.get("temperature", 0.0),
            top_k=model_parameters.get("top_k", 5),
            top_p=model_parameters.get("top_p", 0.1),
            seed=model_parameters.get("seed", 48),
            max_output_tokens=4096,
        )
    if provider == "ollama":
        logging.info("Loading Ollama LLM")
        download_ollama_model(model)

        ollama_kwargs = {
            "model": model,
            "temperature": model_parameters.get("temperature", 0.0),
            "top_k": model_parameters.get("top_k", 5),
            "top_p": model_parameters.get("top_p", 0.1),
            "seed": model_parameters.get("seed", 48),
            "frequency_penalty": model_parameters.get("frequency_penalty", 0.15),
            "presence_penalty": model_parameters.get("presence_penalty", 0.1),
            "num_ctx": 32768,
            "num_predict": 4096,
        }

        if model_parameters.get("format") == "json":
            ollama_kwargs["format"] = "json"

        return ChatOllama(**ollama_kwargs, client_kwargs={"timeout": 300.0})
    else:
        raise ValueError(f"Provider {provider} nicht unterstützt.")


def run_experiment():
    try:
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
            choices=["google", "ollama"],
            default="ollama",
            help="LLM provider to use for conditions C2-C4",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="llama3",
            help="Name of the Ollama model to use",
        )
        parser.add_argument(
            "--log",
            type=str,
            default="INFO",
            choices=["INFO", "DEBUG"],
            help="Puts the logger in DEBUG or INFO mode",
        )

        args = parser.parse_args()

        setup_logging(args.log, args.experiment, args.model, args.provider)
        logging.info("Starting Document Automation Benchmark Experiment")

        evaluator = BenchmarkEvaluator(
            results_dir=str(PROJECT_ROOT / "results"),
            experiment=args.experiment,
            complexity=args.complexity,
        )
        data_dir_path = str(PROJECT_ROOT / "data" / "processed")
        loader = DataLoader(data_dir=data_dir_path, experiment=args.experiment)

        base_llm_params = {
            "model": args.model,
            "temperature": 0.0,
            "seed": 48,
            "top_k": 3,
            "top_p": 0.1,
            "num_ctx": 32768,
            "frequency_penalty": 0.15,
            "presence_penalty": 0.1,
            "num_predict": 4096,
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
                logging.debug(doc)

                prediction: dict = {}
                meta_data: dict = {}
                error_msg = None

                start = perf_counter()
                try:
                    # error_msg is None when succesfull
                    prediction, meta_data, error_msg = condition_instance.extract_data(
                        doc
                    )
                except Exception as e:
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

            # Send to sleep to not hit the rate limit of the Google Gemini API
            if args.provider == "google":
                logging.info("Sleeping for 15 seconds to avoid rate limit.")
                time.sleep(15)

        logging.info("Experiment completed successfully.")

    except Exception as e:
        logging.error(f"Experiment execution failed: {e}")
        raise e
