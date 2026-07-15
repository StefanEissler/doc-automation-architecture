import datetime as dt
import logging
import csv
import json
from pathlib import Path
from typing import Any, Union, List
import re

import dateutil.parser
from rapidfuzz import fuzz

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
    "AddressMatch": general_string_cleaner,
    "NumericalStringMatch": lambda x: re.sub(r"[^\d]", "", str(x)),
}


# Vergleichsfunktionen für verschiedene Matching-Methoden
def compare_exact(gt_norm: Any, pred_norm: Union[Any, List[Any]]) -> bool:
    if gt_norm is None:
        return False
    if isinstance(pred_norm, list):
        return any(str(gt_norm) == str(p) for p in pred_norm if p is not None)
    return str(gt_norm) == str(pred_norm) if pred_norm is not None else False


def compare_substring(gt_norm: Any, pred_norm: Union[Any, List[Any]]) -> bool:
    if gt_norm is None:
        return False
    gt_str = str(gt_norm)
    if isinstance(pred_norm, list):
        return any(gt_str in str(p) for p in pred_norm if p is not None)
    return gt_str in str(pred_norm) if pred_norm is not None else False


def compare_fuzzy(
    gt_norm: Any, pred_norm: Union[Any, List[Any]], threshold: float = 80.0
) -> bool:
    if gt_norm is None:
        return False
    gt_str = str(gt_norm)
    if isinstance(pred_norm, list):
        return any(
            fuzz.ratio(gt_str, str(p)) >= threshold for p in pred_norm if p is not None
        )
    return (
        fuzz.ratio(gt_str, str(pred_norm)) >= threshold
        if pred_norm is not None
        else False
    )


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

    def evaluate_line_items(
        self, gt_items: list, pred_items: list, method: str = "fuzzy"
    ) -> dict:
        """Evaluates line items matching by parsing dictionaries key by key."""
        tp, fp, fn = 0, 0, 0
        if not gt_items:
            fp = len(pred_items) if pred_items else 0
            return {"tp": tp, "fp": fp, "fn": fn}
        if not pred_items:
            fn = len(gt_items)
            return {"tp": tp, "fp": fp, "fn": fn}

        matched_gt_indices = set()

        for pred_item in pred_items:
            best_match_idx = -1

            for i, gt_item in enumerate(gt_items):
                if i in matched_gt_indices:
                    continue

                is_match = True

                if isinstance(gt_item, dict) and isinstance(pred_item, dict):
                    for key, gt_val in gt_item.items():
                        pred_val = pred_item.get(key)

                        match_func_name = DEEPFORM_META[
                            "entity_name_to_match_func"
                        ].get(key, "GeneralStringMatch")
                        cleaning_func = CLEANING_FUNCTIONS.get(
                            match_func_name, general_string_cleaner
                        )

                        clean_gt = cleaning_func(gt_val) if gt_val else None
                        clean_pred = cleaning_func(pred_val) if pred_val else None

                        if not clean_gt and not clean_pred:
                            continue
                        if clean_gt and not clean_pred:
                            is_match = False
                            break

                        if method == "exact" and not compare_exact(
                            clean_gt, clean_pred
                        ):
                            is_match = False
                            break
                        elif method == "substring" and not compare_substring(
                            clean_gt, clean_pred
                        ):
                            is_match = False
                            break
                        elif method == "fuzzy" and not compare_fuzzy(
                            clean_gt, clean_pred
                        ):
                            is_match = False
                            break
                else:
                    # Fallback für reine Strings
                    gt_str = general_string_cleaner(str(gt_item))
                    pred_str = general_string_cleaner(str(pred_item))
                    if method == "exact" and not compare_exact(gt_str, pred_str):
                        is_match = False
                    elif method == "substring" and not compare_substring(
                        gt_str, pred_str
                    ):
                        is_match = False
                    elif method == "fuzzy" and not compare_fuzzy(gt_str, pred_str):
                        is_match = False

                if is_match:
                    best_match_idx = i
                    break

            if best_match_idx != -1:
                matched_gt_indices.add(best_match_idx)
                tp += 1
            else:
                fp += 1

        fn = len(gt_items) - len(matched_gt_indices)
        return {"tp": tp, "fp": fp, "fn": fn}

    def evaluate_field(
        self, target_field: str, ground_truth: Any, predicted: Any, raw_text: str
    ) -> dict:
        """Evaluates a single unrepeated field across exact, substring, and fuzzy matching."""
        match_func_name = DEEPFORM_META["entity_name_to_match_func"].get(
            target_field, "GeneralStringMatch"
        )
        cleaning_func = CLEANING_FUNCTIONS.get(match_func_name, general_string_cleaner)

        gt_norm = cleaning_func(ground_truth) if ground_truth else None

        if isinstance(predicted, dict):
            pred_norm = [cleaning_func(v) for v in predicted.values() if v]
        elif isinstance(predicted, list):
            pred_norm = [cleaning_func(v) for v in predicted if v]
        else:
            pred_norm = cleaning_func(predicted) if predicted else None

        if not gt_norm and not pred_norm:
            return {"exact": "TN", "substring": "TN", "fuzzy": "TN"}
        if not gt_norm and pred_norm:
            return {
                "exact": "FP_HALLUCINATION",
                "substring": "FP_HALLUCINATION",
                "fuzzy": "FP_HALLUCINATION",
            }
        if gt_norm and not pred_norm:
            return {"exact": "FN", "substring": "FN", "fuzzy": "FN"}

        results = {"exact": "FP", "substring": "FP", "fuzzy": "FP"}

        if compare_exact(gt_norm, pred_norm):
            results["exact"] = "TP"
        if compare_substring(gt_norm, pred_norm):
            results["substring"] = "TP"
        if compare_fuzzy(gt_norm, pred_norm, threshold=80.0):
            results["fuzzy"] = "TP"

        content_clean = re.sub(r"\s+", "", str(raw_text).lower())
        for method in ["exact", "substring", "fuzzy"]:
            if results[method] == "FP":
                is_ext_error = False
                if isinstance(pred_norm, list):
                    for p in pred_norm:
                        if p and re.sub(r"\s+", "", str(p)) in content_clean:
                            is_ext_error = True
                            break
                else:
                    if (
                        pred_norm
                        and re.sub(r"\s+", "", str(pred_norm)) in content_clean
                    ):
                        is_ext_error = True

                if is_ext_error:
                    results[method] = "FP_EXTRACTION_ERROR"
                else:
                    results[method] = "FP_HALLUCINATION"

        return results

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

        overall_metrics = {
            "exact": {"tp": 0, "fp": 0, "fn": 0},
            "substring": {"tp": 0, "fp": 0, "fn": 0},
            "fuzzy": {"tp": 0, "fp": 0, "fn": 0},
        }

        schema_match_results = {}
        field_metrics_detailed = {
            method: {field: {"tp": 0, "fp": 0, "fn": 0} for field in target_fields}
            for method in ["exact", "substring", "fuzzy"]
        }

        is_hallucination = False
        is_numeric_hallucination = False
        is_text_hallucination = False

        for field in target_fields:
            gt_val = ground_truth_data.get(field)
            pred_val = predicted_data.get(field)

            result_dict = self.evaluate_field(field, gt_val, pred_val, doc_text)
            schema_match_results[field] = result_dict

            for method in ["exact", "substring", "fuzzy"]:
                res = result_dict[method]
                if res == "TP":
                    overall_metrics[method]["tp"] += 1
                    field_metrics_detailed[method][field]["tp"] += 1
                elif res == "FN":
                    overall_metrics[method]["fn"] += 1
                    field_metrics_detailed[method][field]["fn"] += 1
                elif res in ["FP_EXTRACTION_ERROR", "FP_HALLUCINATION"]:
                    overall_metrics[method]["fp"] += 1
                    field_metrics_detailed[method][field]["fp"] += 1

                    if (
                        res == "FP_HALLUCINATION" and method == "fuzzy"
                    ):  # Baseline trigger
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

        gt_line_items = ground_truth_data.get("line_items", [])
        pred_line_items = predicted_data.get("line_items", [])

        line_item_results = {}
        for method in ["exact", "substring", "fuzzy"]:
            li_res = self.evaluate_line_items(gt_line_items, pred_line_items, method)
            line_item_results[method] = li_res
            overall_metrics[method]["tp"] += li_res["tp"]
            overall_metrics[method]["fp"] += li_res["fp"]
            overall_metrics[method]["fn"] += li_res["fn"]

        def calc_f1(tp, fp, fn):
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            return round(prec, 4), round(rec, 4), round(f1, 4)

        # Calculate final aggregated metrics
        f1_exact = calc_f1(**overall_metrics["exact"])
        f1_substring = calc_f1(**overall_metrics["substring"])
        f1_fuzzy = calc_f1(**overall_metrics["fuzzy"])

        duration_seconds = (
            round(duration, 3) if duration is not None else metadata.get("duration")
        )

        result_entry = {
            "condition": condition_id,
            "complexity": complexity_level,
            "doc_id": doc_id,
            # differenzierte Scores
            "f1_exact": f1_exact[2],
            "prec_exact": f1_exact[0],
            "rec_exact": f1_exact[1],
            "f1_substring": f1_substring[2],
            "prec_substring": f1_substring[0],
            "rec_substring": f1_substring[1],
            "f1_fuzzy": f1_fuzzy[2],
            "prec_fuzzy": f1_fuzzy[0],
            "rec_fuzzy": f1_fuzzy[1],
            "is_hallucination": is_hallucination,
            "is_numeric_hallucination": is_numeric_hallucination,
            "is_text_hallucination": is_text_hallucination,
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
            "all_tokens": metadata.get("tokens"),
            "retries_used": metadata.get("retries_used", 0),
            # "tools_used": metadata.get("used_tools", []),
            "duration_seconds": duration_seconds,
            "match_results_fields": json.dumps(
                schema_match_results, ensure_ascii=False
            ),
            "match_results_line_items": json.dumps(
                line_item_results, ensure_ascii=False
            ),
            "ground_truth": json.dumps(ground_truth_data, ensure_ascii=False),
            "predicted": json.dumps(predicted_data, ensure_ascii=False),
            "model": model,
            "status": "ERROR" if error else "OK",
            "error_message": str(error) if error else "",
        }
        self._append_to_csv(result_entry)

    def _append_to_csv(self, result_entry: dict):
        file_exists = self.csv_filename.exists()
        with open(self.csv_filename, "a", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=result_entry.keys())
            if not file_exists or not self.header_written:
                dict_writer.writeheader()
                self.header_written = True
            dict_writer.writerow(result_entry)
