from src.data_loader import DataLoader
import datetime as dt


def run_condition_c1(complexity, limit, evaluator):
    print("Running rule based.")
    loader = DataLoader(data_dir="data/corpus")
    documents = loader.load_docs(complexity=complexity, limit=limit)

    for doc in documents:
        start_time = dt.time()
        prediction = apply_rule_based_logic(doc)
        duration = dt.time() - start_time

        evaluator.evaluate(
            condition_id="c1_rule_based",
            complexity_level=doc.complexity,
            doc_id=doc.id,
            predicted_data=prediction,
            metadata={"tokens": 0, "duration": duration},  # Regex kostet keine Token
        )


def apply_rule_based_logic(document):
    # Hier implementieren wir die eigentliche Logik, z.B. Regex-Patterns, RPA-Skripte, etc.
    # Das ist natürlich stark abhängig von der Dokumentenart und den zu extrahierenden Informationen.
    # Für dieses Beispiel nehmen wir an, dass wir bestimmte Felder extrahieren wollen.

    extracted_data = {
        "field1": "extracted_value1",
        "field2": "extracted_value2",
    }
    return extracted_data
