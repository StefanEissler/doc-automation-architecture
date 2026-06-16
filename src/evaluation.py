from curses import meta
import datetime as dt
import logging
import csv
import json
from pathlib import Path
from typing import Any
import re

from fuzzywuzzy import fuzz

DEEPFORM_META = {
    "dataset_name": "DeepForm",
    "entity_name_to_match_func": {
        "advertiser": "GeneralStringMatch",
        "agency": "GeneralStringMatch",
        "contract_num": "NumericalStringMatch",
        "flight_from": "DateMatch",
        "flight_to": "DateMatch",
        "gross_amount": "PriceMatch",
        "product": "GeneralStringMatch",
        "tv_address": "AddressMatch",
        "property": "GeneralStringMatch",
        "channel": "GeneralStringMatch",
        "program_desc": "GeneralStringMatch",
        "program_start_date": "DateMatch",
        "program_end_date": "DateMatch",
        "sub_amount": "PriceMatch",
    },
    "entity_appearance_pattern": {
        "advertiser": "unrepeated",
        "agency": "unrepeated",
        "contract_num": "unrepeated",
        "flight_from": "unrepeated",
        "flight_to": "unrepeated",
        "gross_amount": "unrepeated",
        "product": "unrepeated",
        "tv_address": "unrepeated",
        "property": "unrepeated",
        "channel": "line_item",
        "program_desc": "line_item",
        "program_start_date": "line_item",
        "program_end_date": "line_item",
        "sub_amount": "line_item",
    },
}


def price_match_cleaner(value: Any) -> float:
    if not value:
        return None
    cleaned = re.sub(r"[^\d,\.-]", "", str(value))
    return float(cleaned) if cleaned else None


def date_match_cleaner(value: Any) -> str:
    return str(value).strip()


def general_string_cleaner(value: Any) -> str:
    return str(value).strip().lower()


CLEANING_FUNCTIONS = {
    "PriceMatch": price_match_cleaner,
    "DateMatch": date_match_cleaner,
    "GeneralStringMatch": general_string_cleaner,
    "NumericalStringMatch": lambda x: re.sub(r"[^\d]", "", str(x)),
}


class BenchmarkEvaluator:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def _load_ground_truth(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Ground Truth file not found at {path}.")
            return {}

    def evaluate(
        self,
        condition_id,
        complexity_level,
        doc_id,
        predicted_data,
        ground_truth_data,
        metadata,
        duration=None,
    ):
        if not isinstance(predicted_data, dict):
            predicted_data = {}

        tp, fp, fn = 0, 0, 0
        is_hallucination = False

        target_fields = [
            field
            for field, pattern in DEEPFORM_META["entity_appearance_pattern"].items()
            if pattern == "unrepeated"
        ]

        for field in target_fields:
            gt_val = ground_truth_data.get(field)
            pred_val = predicted_data.get(field)

            for _ in predicted_data.keys() - ground_truth_data.keys():
                is_hallucination = True

            result = self.evaluate_field(field, gt_val, pred_val)

            if result == "TP":
                tp += 1
            elif result == "FN":
                fn += 1
            elif result.startswith("FP"):
                fp += 1
                # konfabulierte Pflichtfelder werden als Halluzination gewertet
                if result == "FP_HALLUCINATION" or result == "FP_WRONG_VALUE":
                    is_hallucination = True

        # Makro F1-Score für das gesamte Dokument berechnen
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result_entry = {
            "condition": condition_id,
            "complexity": complexity_level,
            "doc_id": doc_id,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "is_hallucination": is_hallucination,
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "all_tokens": metadata.get("tokens", 0),
            "duration_seconds": (
                (
                    round(metadata.get("duration", 0.0), 3)
                    if duration is not None
                    else metadata.get("duration", 0.0)
                ),
            ),
        }
        self.results.append(result_entry)

    def evaluate_field(
        self, target_field: str, ground_truth: Any, predicted: Any
    ) -> str:
        """Klassifiziert die vorhersage für ein Feld als TP, FP oder FN

        Args:
            target_field (str): _description_
            ground_truth (Any): _description_
            predicted (Any): _description_

        Returns:
            str: _description_
        """

        pattern = DEEPFORM_META["entity_appearance_pattern"].get(target_field)

        if pattern == "line_item":
            return "SKIPPED"

        match_func_name = DEEPFORM_META["entity_name_to_match_func"].get(
            target_field, "GeneralStringMatch"
        )
        cleaning_func = CLEANING_FUNCTIONS.get(match_func_name, general_string_cleaner)

        gt_norm = cleaning_func(ground_truth) if ground_truth else None
        pred_norm = cleaning_func(predicted) if predicted else None

        if not gt_norm and not pred_norm:
            return "TN"
        if not gt_norm and pred_norm:
            return "FP_HALLUCInATION"
        if gt_norm and not pred_norm:
            return "FN"

        if match_func_name in ["PriceMatch", "NumericalStringMatch", "DateMatch"]:
            is_match = gt_norm == pred_norm
        else:
            # Textfelder nutzen Fuzzy Matching (Threshold 80% wie im LayIE-Paper)
            is_match = fuzz.ratio(str(gt_norm), str(pred_norm)) >= 80

        return "TP" if is_match else "FP_WRONG_VALUE"

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
