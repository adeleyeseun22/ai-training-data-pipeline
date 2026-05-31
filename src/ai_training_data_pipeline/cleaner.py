import pandas as pd


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_text_column(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df["text"] = cleaned_df["text"].apply(clean_text)
    return cleaned_df


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df["label"] = (
        cleaned_df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return cleaned_df


def normalize_split_column(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df["split"] = (
        cleaned_df["split"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return cleaned_df


def normalize_source_column(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df["source"] = (
        cleaned_df["source"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )
    return cleaned_df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    cleaned_df["record_id"] = (
        cleaned_df["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned_df = normalize_text_column(cleaned_df)
    cleaned_df = normalize_label_column(cleaned_df)
    cleaned_df = normalize_split_column(cleaned_df)
    cleaned_df = normalize_source_column(cleaned_df)

    return cleaned_df