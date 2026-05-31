from pathlib import Path

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    required_columns: list[str] = Field(
        default_factory=lambda: [
            "record_id",
            "text",
            "label",
            "source",
            "split",
        ]
    )
    valid_labels: set[str] = Field(
        default_factory=lambda: {"positive", "negative", "neutral"}
    )
    valid_splits: set[str] = Field(
        default_factory=lambda: {"train", "validation", "test"}
    )


class PipelineSummary(BaseModel):
    input_file: str
    output_dir: str
    raw_records: int
    clean_records: int
    rejected_records: int
    duplicate_record_ids: int
    missing_record_ids: int
    missing_text_records: int
    invalid_label_records: int
    invalid_split_records: int
    train_records: int
    validation_records: int
    test_records: int
    label_distribution: dict[str, int]


class PipelineOutputs(BaseModel):
    clean_csv_path: Path
    clean_jsonl_path: Path
    rejected_csv_path: Path
    summary_json_path: Path
    summary: PipelineSummary