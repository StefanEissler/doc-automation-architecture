import datetime as dt
import logging
import csv
import json
from pathlib import Path
from typing import Any
import re

import dateutil.parser
from rapidfuzz import fuzz  # Upgrade von fuzzywuzzy auf rapidfuzz

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

    val_str = str(value)
    cleaned = re.sub(r"[^\d,\.-]", "", val_str)
    if not cleaned:
        return None

    if "." in cleaned and "," in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        if len(cleaned.split(",")[-1]) == 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def date_match_cleaner(value: Any) -> str:
    if not value:
        return None
    val_str = str(value).strip()
    try:
        parsed_date = dateutil.parser.parse(val_str, fuzzy=True)
        return parsed_date.strftime("%Y-%m-%d")
    except Exception:
        return val_str.lower()


def general_string_cleaner(value: Any) -> str:
    return str(value).strip().lower()


CLEANING_FUNCTIONS = {
    "PriceMatch": price_match_cleaner,
    "DateMatch": date_match_cleaner,
    "GeneralStringMatch": general_string_cleaner,
    "NumericalStringMatch": lambda x: re.sub(r"[^\d]", "", str(x)),
}


class BenchmarkEvaluator:

    def __init__(self, results_dir: str, experiment: str, complexity: Any = "All"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

        comp_str = "-".join(complexity) if isinstance(complexity, list) else complexity
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = (
            self.results_dir / f"experiment_benchmark_{experiment}_{comp_str}_{now}.csv"
        )
        self.header_written = False

    def _load_ground_truth(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Ground Truth file not found at {path}.")
            return {}

    def _compare_exact(self, gt_norm: Any, pred_norm: Any) -> bool:
        if isinstance(pred_norm, (list, tuple)):
            return any(gt_norm == p for p in pred_norm)
        return gt_norm == pred_norm

    def _compare_fuzzy(
        self, gt_norm: str, pred_norm: Any, threshold: float = 80.0
    ) -> bool:
        gt_str = str(gt_norm)
        if isinstance(pred_norm, (list, tuple)):
            return any(fuzz.ratio(gt_str, str(p)) >= threshold for p in pred_norm)
        return fuzz.ratio(gt_str, str(pred_norm)) >= threshold

    def evaluate(
        self,
        condition_id: str,
        complexity_level: str,
        doc_id: str,
        predicted_data: dict,
        ground_truth_data: dict,
        doc_text: str,
        metadata: dict = None,
        duration: float = None,
        model: str = None,
        error: Exception = None,
    ):
        if not isinstance(predicted_data, dict):
            predicted_data = {}
        if metadata is None:
            metadata = {}

        target_fields = [
            field
            for field, pattern in DEEPFORM_META["entity_appearance_pattern"].items()
            if pattern == "unrepeated"
        ]

        # Tracking fields and metrics
        schema_metrics = {}
        schema_match_results = {}

        is_hallucination = False
        is_numeric_hallucination = False
        is_text_hallucination = False

        total_tp, total_fp, total_fn = 0, 0, 0

        # field specific evaluation
        for field in target_fields:
            gt_val = ground_truth_data.get(field)
            pred_val = predicted_data.get(field)

            result_class = self.evaluate_field(field, gt_val, pred_val, doc_text)
            schema_match_results[field] = result_class

            # Initalise counter for the field if not already present
            schema_metrics[field] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

            if result_class == "TP":
                schema_metrics[field]["tp"] += 1
                total_tp += 1
            elif result_class == "FN":
                schema_metrics[field]["fn"] += 1
                total_fn += 1
            elif result_class in ["FP_EXTRACTION_ERROR", "FP_HALLUCINATION"]:
                schema_metrics[field]["fp"] += 1
                total_fp += 1

                if result_class == "FP_HALLUCINATION":
                    is_hallucination = True
                    match_func = DEEPFORM_META["entity_name_to_match_func"].get(
                        field, "GeneralStringMatch"
                    )
                    if match_func in [
                        "PriceMatch",
                        "NumericalStringMatch",
                        "DateMatch",
                    ]:
                        is_numeric_hallucination = True
                    else:
                        is_text_hallucination = True

            # Count field metrics for precision, recall, and F1-score
            tp_f = schema_metrics[field]["tp"]
            fp_f = schema_metrics[field]["fp"]
            fn_f = schema_metrics[field]["fn"]

            if (tp_f + fp_f + fn_f) == 0:
                schema_metrics[field]["precision"] = None
                schema_metrics[field]["recall"] = None
                schema_metrics[field]["f1"] = None
            else:
                prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
                rec_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
                f1_f = (
                    2 * (prec_f * rec_f) / (prec_f + rec_f)
                    if (prec_f + rec_f) > 0
                    else 0.0
                )

                schema_metrics[field]["precision"] = prec_f
                schema_metrics[field]["recall"] = rec_f
                schema_metrics[field]["f1"] = f1_f

        # Calcuate overall metrics per document
        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0.0
        )

        duration_seconds = (
            round(duration, 3)
            if duration is not None
            else (
                round(metadata.get("duration", 0), 3)
                if metadata.get("duration") is not None
                else None
            )
        )

        # Wide format result entry for CSV output
        result_entry = {
            "condition": condition_id,
            "complexity": complexity_level,
            "doc_id": doc_id,
            "overall_f1_score": overall_f1,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "is_hallucination": is_hallucination,
            "is_numeric_hallucination": is_numeric_hallucination,
            "is_text_hallucination": is_text_hallucination,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_target_fields": len(target_fields),
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
            "all_tokens": metadata.get("tokens"),
            "retries_used": metadata.get("retries_used", 0),
            "duration_seconds": duration_seconds,
            "field_metrics": json.dumps(schema_metrics, ensure_ascii=False),
            "match_results": json.dumps(schema_match_results, ensure_ascii=False),
            "ground_truth": json.dumps(ground_truth_data, ensure_ascii=False),
            "predicted": json.dumps(predicted_data, ensure_ascii=False),
            "model": model,
            "status": "ERROR" if error else "OK",
            "error_message": str(error) if error else "",
        }
        self._append_to_csv(result_entry)

    def evaluate_field(
        self, target_field: str, ground_truth: Any, predicted: Any, raw_text: str
    ) -> str:
        pattern = DEEPFORM_META["entity_appearance_pattern"].get(target_field)
        if pattern == "line_item":
            return "SKIPPED"

        match_func_name = DEEPFORM_META["entity_name_to_match_func"].get(
            target_field, "GeneralStringMatch"
        )
        cleaning_func = CLEANING_FUNCTIONS.get(match_func_name, general_string_cleaner)

        gt_norm = cleaning_func(ground_truth) if ground_truth else None
        is_schema_violation = isinstance(predicted, (dict, list))
        pred_norm = (
            None
            if is_schema_violation
            else (cleaning_func(predicted) if predicted else None)
        )

        if not gt_norm and not pred_norm and not is_schema_violation:
            return "TN"
        if not gt_norm and (pred_norm or is_schema_violation):
            return "FP_HALLUCINATION"
        if gt_norm and not pred_norm and not is_schema_violation:
            return "FN"

        if not is_schema_violation:
            if match_func_name in ["PriceMatch", "NumericalStringMatch", "DateMatch"]:
                is_match = self._compare_exact(gt_norm, pred_norm)
            else:
                is_match = self._compare_fuzzy(gt_norm, pred_norm, threshold=80.0)

            if is_match:
                return "TP"

        if is_schema_violation:
            if isinstance(predicted, dict):
                pred_search_strings = [str(v).lower() for v in predicted.values() if v]
            else:
                pred_search_strings = [str(v).lower() for v in predicted if v]
        else:
            pred_search_strings = [str(predicted).lower()]

        content_clean = re.sub(r"\s+", "", str(raw_text).lower())

        for search_str in pred_search_strings:
            pred_clean = re.sub(r"\s+", "", search_str)
            if not pred_clean:
                continue

            if pred_clean in content_clean:
                return "FP_EXTRACTION_ERROR"

            # Substring Match via RapidFuzz
            if fuzz.partial_ratio(pred_clean, content_clean) >= 90.0:
                return "FP_EXTRACTION_ERROR"

            if match_func_name in ["PriceMatch", "NumericalStringMatch"]:
                pred_digits = re.sub(r"[^\d]", "", search_str)
                doc_digits = re.sub(r"[^\d]", "", str(raw_text))
                if pred_digits and pred_digits in doc_digits:
                    return "FP_EXTRACTION_ERROR"

        return "FP_HALLUCINATION"

    def _append_to_csv(self, result_entry: dict):
        """Hängt ein einzelnes Ergebnis an die CSV an (erstellt Header falls Datei neu)."""
        file_exists = self.csv_filename.exists()

        with open(self.csv_filename, "a", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=result_entry.keys())

            if not file_exists or not self.header_written:
                dict_writer.writeheader()
                self.header_written = True

            dict_writer.writerow(result_entry)

        logging.debug(
            f"Result safed to {self.csv_filename.name} with id: {result_entry['doc_id']}"
        )
