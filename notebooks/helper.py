def transform_to_standard_format(
    df,
    complexity_label,
    target_fields,
    raw_ocr_text,
    final_documents,
    source="VRDU_ad-buy-form",
):
    for idx, row in df.iterrows():
        doc_entry = {
            "id": f"ad_buy_{complexity_label}_{idx}",
            "complexity": complexity_label,
            "content": raw_ocr_text,
            "target_fields": target_fields,
            "metadata": {
                "source": source,
                "vendor_name": row.get("vendor_name", "unknown"),
                "regex_patterns": (
                    row.get("regex_patterns", {}) if complexity_label == "L1" else {}
                ),
            },
            "ground_truth": row["ground_truth_clean"],
        }
        final_documents.append(doc_entry)
