import json
from pathlib import Path
from typing import Any

import pandas as pd

from ai_training_data_pipeline.models import PipelineSummary


def model_to_dict(model: PipelineSummary) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_jsonl(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def save_summary_json(summary: PipelineSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(model_to_dict(summary), indent=2),
        encoding="utf-8",
    )