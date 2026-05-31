from pathlib import Path

import pandas as pd

from ai_training_data_pipeline.cleaner import clean_training_data
from ai_training_data_pipeline.pipeline import run_pipeline
from ai_training_data_pipeline.validator import identify_invalid_records


def build_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
                "R003",
                "R004",
                "R005",
                "R006",
                "R006",
                "R007",
                "R008",
                "R009",
                "R010",
            ],
            "text": [
                "This response is helpful and clear.",
                "Terrible answer with wrong information.",
                "Average response, not very detailed.",
                "",
                "This is excellent!",
                "Bad output",
                "Bad output",
                "I am not sure about this answer.",
                "   Helpful but too short.   ",
                None,
                "Neutral and factual response.",
            ],
            "label": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "Positive",
                "NEGATIVE",
                "NEGATIVE",
                "unclear",
                "positive",
                "negative",
                "neutral",
            ],
            "source": [
                "annotation_batch_1",
                "annotation_batch_1",
                "annotation_batch_1",
                "annotation_batch_1",
                "annotation_batch_2",
                "annotation_batch_2",
                "annotation_batch_2",
                "annotation_batch_2",
                "annotation_batch_3",
                "annotation_batch_3",
                "annotation_batch_3",
            ],
            "split": [
                "train",
                "train",
                "validation",
                "train",
                "train",
                "test",
                "test",
                "validation",
                "train",
                "test",
                "validation",
            ],
        }
    )


def test_clean_training_data_normalizes_fields() -> None:
    df = build_sample_dataframe()

    cleaned_df = clean_training_data(df)

    assert cleaned_df.loc[4, "label"] == "positive"
    assert cleaned_df.loc[5, "label"] == "negative"
    assert cleaned_df.loc[8, "text"] == "Helpful but too short."


def test_identify_invalid_records_flags_bad_rows() -> None:
    df = build_sample_dataframe()
    cleaned_df = clean_training_data(df)

    rejected_df = identify_invalid_records(cleaned_df)

    assert len(rejected_df) == 4
    assert "rejection_reasons" in rejected_df.columns
    assert rejected_df["rejection_reasons"].str.len().gt(0).all()


def test_run_pipeline_exports_outputs(tmp_path: Path) -> None:
    df = build_sample_dataframe()
    input_file = tmp_path / "raw_training_data.csv"
    output_dir = tmp_path / "processed"

    df.to_csv(input_file, index=False)

    outputs = run_pipeline(input_file=input_file, output_dir=output_dir)

    assert outputs.clean_csv_path.exists()
    assert outputs.clean_jsonl_path.exists()
    assert outputs.rejected_csv_path.exists()
    assert outputs.summary_json_path.exists()

    assert outputs.summary.raw_records == 11
    assert outputs.summary.clean_records == 7
    assert outputs.summary.rejected_records == 4
    assert outputs.summary.train_records == 4
    assert outputs.summary.validation_records == 2
    assert outputs.summary.test_records == 1