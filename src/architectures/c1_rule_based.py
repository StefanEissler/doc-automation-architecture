import re
import logging
from typing import Optional, Tuple, Dict, List, Any


class RuleBasedCondition:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Regex Pattern für Ad-buy Formulare
        self.header_patterns = {
            "advertiser": r"Advertiser[\s:]+(?!Agency|Buyer|Salesperson|Product|Estimate)((?:\S+\s+){2,6})",
            "agency": r"Agency[\s:]+(?!Buyer|Salesperson|Product)((?:\S+\s+){2,6})",
            "contract_num": r"(?:Contract|Order)(?:\s*(?:#|Num|Number))?[\s:]*([0-9]{5,10})",
            "flight_from": r"(?:Dates|Period|Flight).*?(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})\s*(?:-|to)",
            "flight_to": r"(?:Dates|Period|Flight).*?\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\s*(?:-|to)\s*(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})",
            "gross_amount": r"(?:Gross Total|Grand Total|Net Total)[\s\w]*?(\$[\d\.,]+)",
            "product": r"(?:Product)[\s:]*([^\n]{1,100}?)(?:\s{2,}|\n|$)",
            "tv_address": r"(?:PO Box|P\.O\. Box)[\s:]*(.{10,50}?\d{5}(?:-\d{4})?)",
            "property": r"(?:REMIT TO|Property|Station)[\s:]*(.{5,30}?)(?:\s{2,}|PO Box|P\.O\.)",
        }

        #  Regex für Line Items der Ad-buy Formulare
        self.line_item_pattern = re.compile(
            r"(?P<program_start_date>\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})"  # Startdatum
            r".{3,80}?"  # Ignoriere 3 bis max 80 Zeichen
            r"(?P<sub_amount>\$[\d\.,]+)"  # Der Preis
        )

    def extract_header(self, text: str) -> Dict[str, str]:
        """Extrahiert einmalig auftretende Metadaten aus dem gesamten Text."""
        extracted = {}
        for field, pattern in self.header_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            extracted[field] = match.group(1).strip() if match else ""
        return extracted

    def extract_line_items(self, text: str) -> List[Dict[str, str]]:
        """Iteriert zeilenweise über den Text, um wiederkehrende Tabellendaten zu finden."""
        line_items = []
        for match in self.line_item_pattern.finditer(text):
            line_items.append(
                {
                    "program_desc": "",
                    "channel": "",
                    "program_start_date": match.group("program_start_date"),
                    "program_end_date": "",
                    "sub_amount": match.group("sub_amount"),
                }
            )
        return line_items

    def extract_data(self, document: Any) -> Tuple[Dict, Dict, Optional[str]]:
        """Hauptmethode für die Bedingung C1."""
        try:
            text = document.content
            extracted_data = self.extract_header(text)
            extracted_data["line_items"] = self.extract_line_items(text)

            self.logger.info(f"Finished C1 for: {getattr(document, 'id', 'unknown')}")

            return (
                extracted_data,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tokens": 0,
                    "duration": 0.0,
                },
                None,
            )

        except Exception as e:
            self.logger.error(f"C1 Error: {e}")
            return (
                {},
                {"input_tokens": 0, "output_tokens": 0, "tokens": 0, "duration": 0.0},
                str(e),
            )
