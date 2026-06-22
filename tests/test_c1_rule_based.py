import pytest
from src.data_loader import Document
from src.architectures.c1_rule_based import RuleBasedCondition


@pytest.fixture
def sample_document():
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
        ground_truth={},
        content=ocr_text,
        target_fields=["advertiser", "property", "gross_amount", "contract_num"],
    )


def test_extract_standard_fields(sample_document):
    """Testet, ob normale Felder gefunden werden und das 3-Tuple korrekt ist."""
    extractor = RuleBasedCondition()
    # NEU: Wir erwarten ein 3-Tuple (Prediction, Metadata, Error)
    result, metadata, error = extractor.extract_data(sample_document)

    assert error is None
    # Der Regex nimmt mehrere Wörter mit, also prüfen wir auf Substrings
    assert "Michael Bloomberg" in result["advertiser"]
    assert "KMSP-TV" in result["property"]
    assert metadata["tokens"] == 0
    assert metadata["input_tokens"] == 0


def test_missing_fields_return_empty_string(sample_document):
    """Testet, ob nicht vorhandene Felder leer zurückgegeben werden."""
    extractor = RuleBasedCondition()
    result, _, _ = extractor.extract_data(sample_document)

    # Im Text steht 'INVOICE #', der Regex für contract_num erfordert aber meist 'Contract'
    # oder 'Order'. Wir nutzen .get(), um einen KeyError zu vermeiden.
    assert result.get("contract_num", "") == ""
