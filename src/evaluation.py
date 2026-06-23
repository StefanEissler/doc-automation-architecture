import datetime as dt
import logging
import csv
import json
from pathlib import Path
from typing import Any
import re

import dateutil
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

    val_str = str(value)
    # Alles entfernen außer Ziffern, Komma, Punkt und Minus (Währungen wie €, $ fliegen raus)
    cleaned = re.sub(r"[^\d,\.-]", "", val_str)
    if not cleaned:
        return None

    if "." in cleaned and "," in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            # EU-Format -> Tausenderpunkt weg, Komma zu Dezimalpunkt
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            # US-Format -> Tausenderkomma weg
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # Nur Komma vorhanden: Entweder "1,234" (US) oder "1234,56" (EU)
        # Heuristik: Wenn genau 2 Ziffern nach dem Komma kommen, ist es zu 99% EU-Dezimal
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
        # fuzzy=True ignoriert störenden Text (z.B. "Date: 07/01/2022")
        parsed_date = dateutil.parser.parse(val_str, fuzzy=True)
        # Normalisierung
        return parsed_date.strftime("%Y-%m-%d")
    except Exception:
        # Fallback auf reinen String, falls das Parsen fehlschlägt
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

    def _compare_exact(self, gt_norm: Any, pred_norm: Any) -> bool:
        """Strikter Vergleich für Zahlen, Preise und Daten"""
        # Fallback auf Listen/Tuple-Behandlung, falls das LLM Arrays generiert
        if isinstance(pred_norm, (list, tuple)):
            return any(gt_norm == p for p in pred_norm)
        return gt_norm == pred_norm

    def _compare_fuzzy(self, gt_norm: str, pred_norm: Any, threshold: int = 80) -> bool:
        """Fuzzy Matching für Textfelder"""
        gt_str = str(gt_norm)

        if isinstance(pred_norm, (list, tuple)):
            return any(fuzz.ratio(gt_str, str(p)) >= threshold for p in pred_norm)

        return fuzz.ratio(gt_str, str(pred_norm)) >= threshold

    def evaluate(
        self,
        condition_id,
        complexity_level,
        doc_id,
        predicted_data,
        ground_truth_data,
        doc_text,
        metadata,
        duration=None,
        model=None,
        error=None,
    ):
        if not isinstance(predicted_data, dict):
            predicted_data = {}

        if metadata is None:
            metadata = {}

        tp, fp, fn = 0, 0, 0

        is_hallucination = False
        is_numeric_hallucination = False
        is_text_hallucination = False

        target_fields = [
            field
            for field, pattern in DEEPFORM_META["entity_appearance_pattern"].items()
            if pattern == "unrepeated"
        ]

        for field in target_fields:
            gt_val = ground_truth_data.get(field)
            pred_val = predicted_data.get(field)

            result = self.evaluate_field(field, gt_val, pred_val, doc_text)

            if result == "TP":
                tp += 1
            elif result == "FN":
                fn += 1
            elif result == "FP_EXTRACTION_ERROR":
                fp += 1
            elif result == "FP_HALLUCINATION":
                fp += 1
                is_hallucination = True
                match_func = DEEPFORM_META["entity_name_to_match_func"].get(
                    field, "GeneralStringMatch"
                )
                if match_func in ["PriceMatch", "NumericalStringMatch", "DateMatch"]:
                    is_numeric_hallucination = True
                else:
                    is_text_hallucination = True

        # Makro F1-Score für das gesamte Dokument berechnen
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if duration is not None:
            duration_seconds = round(duration, 3)
        else:
            meta_duration = metadata.get("duration")
            duration_seconds = (
                round(meta_duration, 3) if meta_duration is not None else None
            )

        total_target_fields = len(target_fields)

        result_entry = {
            "condition": condition_id,
            "complexity": complexity_level,
            "doc_id": doc_id,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "is_hallucination": is_hallucination,
            "is_numeric_hallucination": is_numeric_hallucination,
            "is_text_hallucination": is_text_hallucination,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_target_fields": total_target_fields,
            "fields": [
                {
                    "field_name": field,
                    "ground_truth": ground_truth_data.get(field),
                    "extracted": predicted_data.get(field),
                }
                for field in target_fields
            ],
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
            "all_tokens": metadata.get("tokens"),
            "duration_seconds": duration_seconds,
            "ground_truth_raw": json.dumps(ground_truth_data, ensure_ascii=False),
            "predicted_raw": json.dumps(predicted_data, ensure_ascii=False),
            "model": model,
            "status": "ERROR" if error else "OK",
            "error_message": str(error) if error else "",
        }
        self.results.append(result_entry)

    def evaluate_field(
        self, target_field: str, ground_truth: Any, predicted: Any, raw_text: str
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
            return "FP_HALLUCINATION"
        if gt_norm and not pred_norm:
            return "FN"

        if match_func_name in ["PriceMatch", "NumericalStringMatch", "DateMatch"]:
            is_match = self._compare_exact(gt_norm, pred_norm)
        else:
            # Textfelder nutzen Fuzzy Matching (Threshold 80% wie im LayIE-Paper)
            is_match = self._compare_fuzzy(gt_norm, pred_norm, threshold=80)

        if is_match:
            return "TP"

        content_clean = re.sub(r"\s+", "", str(raw_text).lower())
        pred_clean = re.sub(r"\s+", "", str(predicted).lower())

        # Exakter Substring Match (Ist die falsche Zahl/Wort exakt so im Text?)
        if pred_clean in content_clean:
            return "FP_EXTRACTION_ERROR"

        # Fuzzy Substring Match für OCR-Fehler (z.B. "100.0O" statt "100.00")
        if fuzz.partial_ratio(pred_clean, content_clean) >= 90:
            return "FP_EXTRACTION_ERROR"

        # Wenn der Wert nicht im Text steht -> Echte Konfabulation
        return "FP_HALLUCINATION"

    def save_to_csv(self, condition, complexity):
        if not self.results:
            logging.warning("No results to save.")
            return

        comp_str = "-".join(complexity) if isinstance(complexity, list) else complexity
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            self.results_dir / f"experiment_benchmark_{condition}_{comp_str}_{now}.csv"
        )

        keys = self.results[0].keys()

        with open(filename, "w", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.results)

        logging.info(f"Successfully saved {len(self.results)} results to {filename}")
