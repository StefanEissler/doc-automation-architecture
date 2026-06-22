# Auto generated and updated
import pytest
import json
from pathlib import Path
from src.evaluation import BenchmarkEvaluator


@pytest.fixture
def evaluator(tmp_path):
    return BenchmarkEvaluator(results_dir=str(tmp_path))


def test_price_exact_match_format_cleaning(evaluator):
    ground_truth = {"gross_amount": "$1,234.56"}
    predicted = {"gross_amount": 1234.56}
    doc_text = "The total gross amount is $1,234.56 today."

    evaluator.evaluate(
        "TEST", "L1", "doc1", predicted, ground_truth, doc_text, metadata=None
    )
    result = evaluator.results[-1]

    assert result["f1_score"] == 1.0
    assert result["is_numeric_hallucination"] is False
    assert result["is_hallucination"] is False


def test_numeric_hallucination_detection(evaluator):
    """
    Testfall 2 (Korrigiert): Prüft, ob eine numerisch FALSCHE Zahl,
    die NICHT im Text steht, als echte Halluzination erkannt wird.
    """
    ground_truth = {"contract_num": "98765"}
    predicted = {"contract_num": "99999"}  # Eine echte, numerische Halluzination
    doc_text = "Your contract number is 98765. Thank you."

    evaluator.evaluate(
        "TEST", "L1", "doc2", predicted, ground_truth, doc_text, metadata=None
    )
    result = evaluator.results[-1]

    assert result["f1_score"] == 0.0
    assert result["is_numeric_hallucination"] is True
    assert result["is_hallucination"] is True


def test_nlp_standard_hallucination_vs_extraction(evaluator):
    """
    NEU: Testet die strikte Trennung von Extraction Errors (Zahl steht im Text)
    und echten Halluzinationen (Zahl ist frei erfunden).
    """
    document_content = (
        "The total gross amount is 500.00, but the net amount was 400.00."
    )
    ground_truth = {"gross_amount": "$500.00"}

    # Fall A: Extraction Error (LLM liest den falschen, aber existierenden Netto-Betrag aus)
    predicted_extraction_error = {"gross_amount": 400.00}
    evaluator.evaluate(
        "C3",
        "L1",
        "doc_A",
        predicted_extraction_error,
        ground_truth,
        document_content,
        metadata=None,
    )
    res_A = evaluator.results[-1]

    assert res_A["f1_score"] == 0.0  # Es ist ein Fehler...
    assert (
        res_A["is_hallucination"] is False
    )  # ... aber KEINE Halluzination, weil 400.00 im Text steht!

    # Fall B: Echte Halluzination (LLM erfindet eine Zahl, die nirgends im Text steht)
    predicted_hallucination = {"gross_amount": 999.99}
    evaluator.evaluate(
        "C3",
        "L1",
        "doc_B",
        predicted_hallucination,
        ground_truth,
        document_content,
        metadata=None,
    )
    res_B = evaluator.results[-1]

    assert res_B["f1_score"] == 0.0  # Es ist ein Fehler...
    assert res_B["is_numeric_hallucination"] is True  # ... UND eine Halluzination!
    assert res_B["is_hallucination"] is True


def test_price_stress_tests_vrdu():
    """
    Härtetest für den price_match_cleaner, angelehnt an die Edge-Cases
    aus dem VRDU-Benchmark (Währungssymbole, EU vs US Formate, Vorzeichen).
    """
    from src.evaluation import price_match_cleaner

    # 1. VRDU Case: Fehlendes Dollarzeichen und Leerzeichen
    assert price_match_cleaner("$ 40,000") == 40000.0
    assert price_match_cleaner("40000.00") == 40000.0

    # 2. Europäisches Format mit Euro-Zeichen
    assert price_match_cleaner("€ 1.234,56") == 1234.56
    assert price_match_cleaner("1.234,56 €") == 1234.56

    # 3. US Format mit extremen Tausender-Kommas
    assert price_match_cleaner("$1,234,567.89") == 1234567.89

    # 4. Negativ-Beträge (Gutschriften)
    assert price_match_cleaner("- 500,00 EUR") == -500.0
    assert price_match_cleaner("-$500.00") == -500.0

    # 5. Wilde String-Halluzinationen des LLMs (Konfabulation mit Text)
    assert price_match_cleaner("Der Betrag ist 150.00") == 150.0


def test_date_cleaning_vrdu_logic():
    from src.evaluation import date_match_cleaner

    # VRDU Beispiel: Unterschiedliche Formate, selbes Datum
    assert date_match_cleaner("July 1, 2022") == date_match_cleaner("07/01/2022")
    # LLM Chattiness
    assert date_match_cleaner("The flight is on 2022-07-01.") == "2022-07-01"


def test_fuzzy_text_match_success(evaluator):
    ground_truth = {"advertiser": "Acme Corp LLC"}
    predicted = {"advertiser": "Acme Corp"}
    doc_text = "Advertiser: Acme Corp LLC"

    evaluator.evaluate(
        "TEST", "L1", "doc3", predicted, ground_truth, doc_text, metadata=None
    )
    result = evaluator.results[-1]

    assert result["f1_score"] == 1.0


def test_real_document_evaluation(evaluator):
    """
    Testfall 4: Lädt ein echtes Dokument aus dem Corpus und prüft
    eine teils korrekte, teils halluzinierte Vorhersage.
    """
    # Pfad zur JSON (passe diesen an, falls du den Test aus einem anderen Ordner startest)
    json_path = Path("data/processed/corpus_experiment_A_internal.json")

    # Überspringe den Test sauber, falls die Datei auf dem System (noch) nicht existiert
    if not json_path.exists():
        pytest.skip(f"Datei {json_path} nicht gefunden.")

    with open(json_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Wir holen uns das spezifische Dokument
    doc_id = "0ea66228-6efd-8386-8000-e2e0c72e3825.pdf"
    doc_data = next((d for d in corpus if d["doc_id"] == doc_id), None)
    assert doc_data is not None, "Test-Dokument nicht im Corpus gefunden."

    ground_truth = doc_data["ground_truth"]
    doc_text = doc_data.get("ocr_text", "")  # Rohtext aus dem Datensatz holen

    # Wir simulieren eine Vorhersage des LLMs:
    # 1. advertiser: Perfekt
    # 2. gross_amount: Numerisch perfekt, Format anders
    # 3. contract_num: Halluzination! (Falsche Zahl)
    predicted = {
        "advertiser": "Friends of Jeff Sessions Senate Co(134550",
        "gross_amount": 560.00,
        "contract_num": "0000000",  # FALSCH (GT ist 4257313)
    }

    evaluator.evaluate(
        condition_id="C4",
        complexity_level=doc_data["complexity"],
        doc_id=doc_id,
        predicted_data=predicted,
        ground_truth_data=ground_truth,
        doc_text=doc_text,
        metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "tokens": 150,
            "duration": 4.5,
        },
    )

    result = evaluator.results[-1]

    # Überprüfungen
    assert result["is_numeric_hallucination"] is True  # Wegen contract_num
    assert result["f1_score"] < 1.0  # Da nicht alle Felder da sind und eines falsch ist

    # Testen, ob die Metadaten sauber übernommen wurden
    assert result["all_tokens"] == 150
