from collections import Counter

import pandas as pd

from ai_training_data_pipeline.models import PipelineConfig


def validate_required_columns(df: pd.DataFrame, config: PipelineConfig) -> None:
    missing_columns = [
        column for column in config.required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def identify_invalid_records(
    df: pd.DataFrame,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    config = config or PipelineConfig()
    validate_required_columns(df, config)

    working_df = df.copy()

    duplicate_mask = working_df["record_id"].duplicated(keep="first")
    missing_record_id_mask = working_df["record_id"].eq("")
    missing_text_mask = working_df["text"].eq("")
    invalid_label_mask = ~working_df["label"].isin(config.valid_labels)
    invalid_split_mask = ~working_df["split"].isin(config.valid_splits)

    rejection_reasons: list[str] = []

    for index in working_df.index:
        reasons: list[str] = []

        if bool(duplicate_mask.loc[index]):
            reasons.append("duplicate_record_id")

        if bool(missing_record_id_mask.loc[index]):
            reasons.append("missing_record_id")

        if bool(missing_text_mask.loc[index]):
            reasons.append("missing_text")

        if bool(invalid_label_mask.loc[index]):
            reasons.append("invalid_label")

        if bool(invalid_split_mask.loc[index]):
            reasons.append("invalid_split")

        rejection_reasons.append("; ".join(reasons))

    rejected_df = working_df.copy()
    rejected_df["rejection_reasons"] = rejection_reasons

    return rejected_df[rejected_df["rejection_reasons"].str.len() > 0]


def count_issue_type(rejected_df: pd.DataFrame, issue_type: str) -> int:
    if rejected_df.empty or "rejection_reasons" not in rejected_df.columns:
        return 0

    return int(
        rejected_df["rejection_reasons"]
        .fillna("")
        .str.contains(issue_type, regex=False)
        .sum()
    )


def summarize_rejection_reasons(rejected_df: pd.DataFrame) -> dict[str, int]:
    reason_counter: Counter[str] = Counter()

    if rejected_df.empty or "rejection_reasons" not in rejected_df.columns:
        return {}

    for reason_text in rejected_df["rejection_reasons"].dropna():
        reasons = [reason.strip() for reason in str(reason_text).split(";")]
        reason_counter.update(reason for reason in reasons if reason)

    return dict(sorted(reason_counter.items()))