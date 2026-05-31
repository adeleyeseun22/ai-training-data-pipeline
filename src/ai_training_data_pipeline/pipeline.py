from pathlib import Path

import pandas as pd

from ai_training_data_pipeline.cleaner import clean_training_data
from ai_training_data_pipeline.exporter import save_csv, save_jsonl, save_summary_json
from ai_training_data_pipeline.loader import load_csv
from ai_training_data_pipeline.models import (
    PipelineConfig,
    PipelineOutputs,
    PipelineSummary,
)
from ai_training_data_pipeline.validator import (
    count_issue_type,
    identify_invalid_records,
    validate_required_columns,
)


def build_summary(
    input_file: str | Path,
    output_dir: str | Path,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> PipelineSummary:
    label_counts = clean_df["label"].value_counts().to_dict()

    return PipelineSummary(
        input_file=str(input_file),
        output_dir=str(output_dir),
        raw_records=int(len(raw_df)),
        clean_records=int(len(clean_df)),
        rejected_records=int(len(rejected_df)),
        duplicate_record_ids=count_issue_type(rejected_df, "duplicate_record_id"),
        missing_record_ids=count_issue_type(rejected_df, "missing_record_id"),
        missing_text_records=count_issue_type(rejected_df, "missing_text"),
        invalid_label_records=count_issue_type(rejected_df, "invalid_label"),
        invalid_split_records=count_issue_type(rejected_df, "invalid_split"),
        train_records=int((clean_df["split"] == "train").sum()),
        validation_records=int((clean_df["split"] == "validation").sum()),
        test_records=int((clean_df["split"] == "test").sum()),
        label_distribution={str(key): int(value) for key, value in label_counts.items()},
    )


def run_pipeline(
    input_file: str | Path,
    output_dir: str | Path = "data/processed",
    config: PipelineConfig | None = None,
) -> PipelineOutputs:
    config = config or PipelineConfig()
    output_path = Path(output_dir)

    raw_df = load_csv(input_file)
    validate_required_columns(raw_df, config)

    normalized_df = clean_training_data(raw_df)
    rejected_df = identify_invalid_records(normalized_df, config)

    clean_df = normalized_df.drop(index=rejected_df.index).reset_index(drop=True)
    rejected_df = rejected_df.reset_index(drop=True)

    clean_csv_path = output_path / "clean_training_data.csv"
    clean_jsonl_path = output_path / "clean_training_data.jsonl"
    rejected_csv_path = output_path / "rejected_records.csv"
    summary_json_path = Path("reports") / "pipeline_summary.json"

    summary = build_summary(
        input_file=input_file,
        output_dir=output_dir,
        raw_df=raw_df,
        clean_df=clean_df,
        rejected_df=rejected_df,
    )

    save_csv(clean_df, clean_csv_path)
    save_jsonl(clean_df, clean_jsonl_path)
    save_csv(rejected_df, rejected_csv_path)
    save_summary_json(summary, summary_json_path)

    return PipelineOutputs(
        clean_csv_path=clean_csv_path,
        clean_jsonl_path=clean_jsonl_path,
        rejected_csv_path=rejected_csv_path,
        summary_json_path=summary_json_path,
        summary=summary,
    )