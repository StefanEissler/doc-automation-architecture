import datetime as dt
import logging
import csv
import json


class BenchmarkEvaluator:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    def _load_ground_truth(self, path: str) -> dict:
        """Loads the ground truth data from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        except FileNotFoundError:
            logging.warning(
                f"Ground Truth file not found at {path}. Proceeding without it for now."
            )
            return {}

    def evaluate(
        self, condition_id, complexity_level, doc_id, predicted_data, metadata
    ):
        """
        Evaluates the predicted data against the ground truth for a given document and condition.
        """
        # 1. Berechne F1-Score für Pflichtfelder
        # 2. Prüfe auf Halluzinationen
        # 3. Erfasse Token-Costs und Zeit aus metadata
        f1 = 0.85  # Platzhalter für berechneten F1-Score\
        hallucinated = False  # Platzhalter für Halluzinationsprüfung

        result_entry = {
            "condition": condition_id,
            "complexity": complexity_level,
            "doc_id": doc_id,
            "f1_score": f1,
            "is_hallucination": hallucinated,
            "token_cost": metadata["tokens"],
            "duration": metadata["duration"],
        }
        self.results.append(result_entry)

    def save_to_csv(self, condition, complexity):
        if not self.results:
            logging.warning("No results to save.")
            return

        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"evaluation_{now}_{condition}_{complexity}.csv"

        keys = self.results[0].keys()

        with open(filename, "w", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.results)

        logging.info(f"Successfully saved {len(self.results)} results to {filename}")

        pass
