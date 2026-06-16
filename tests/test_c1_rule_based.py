import pytest
from src.data_loader import Document
from src.architectures.c1_rule_based import RuleBasedCondition


@pytest.fixture
def sample_document():
    """
    Ein Pytest-Fixture stellt uns ein standardisiertes Dokument
    für alle unsere Tests zur Verfügung.
    """
    ocr_text = """
    INVOICE # 950658
    ADVERTISER: Michael Bloomberg 2020, Inc
    Property KMSP-TV Fox9
    Gross_Amount = $5,625.00
    Product: MIKE BLOOMBERG FOR SENATE
    """
    return Document(
        id="test_doc_01",
        complexity="L1",
        content=ocr_text,
        metadata={},
        target_fields=[
            "advertiser",
            "property",
            "gross_amount",
            "contract_num",
            "missing_field",
        ],
    )


def test_extract_standard_fields(sample_document):
    """Testet, ob normale Felder (ohne Unterstrich) gefunden werden."""
    extractor = RuleBasedCondition()
    result, tokens = extractor.extract_data(sample_document)

    # Assert prüft, ob eine Bedingung True ist. Wenn nicht, schlägt der Test fehl.
    assert result["advertiser"] == "Michael"
    assert result["property"] == "KMSP-TV"
    assert tokens == 0


def test_extract_underscore_fields(sample_document):
    """Testet, ob Felder mit Unterstrich durch die Fallbacks korrekt gefunden werden."""
    extractor = RuleBasedCondition()
    result, _ = extractor.extract_data(sample_document)

    assert result["gross_amount"] == "$5,625.00"


def test_missing_fields_return_empty_string(sample_document):
    """Testet, ob nicht vorhandene Felder einen leeren String zurückgeben."""
    extractor = RuleBasedCondition()
    result, _ = extractor.extract_data(sample_document)

    assert (
        result["contract_num"] == ""
    )  # Im Text steht 'Invoice #', nicht 'contract num'
    assert result["missing_field"] == ""
