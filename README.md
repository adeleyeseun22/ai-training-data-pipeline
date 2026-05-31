# AI Training Data Pipeline

A Python ETL pipeline for cleaning, validating, deduplicating, and exporting AI training datasets.

This project demonstrates production-style data engineering for AI workflows, including raw data ingestion, schema validation, text cleaning, label normalization, bad-record handling, structured exports, and pipeline summary reporting.

---

## Why This Project Matters

AI models depend on clean, structured, and reliable training data. Raw annotation or training datasets often contain duplicate records, missing text, inconsistent labels, invalid splits, and records that should not be used for model training.

This project provides a lightweight Python ETL pipeline that transforms raw AI training data into clean, training-ready outputs while preserving rejected records for review and audit.

---

## Features

- Load raw training data from CSV
- Validate required dataset columns
- Clean and trim text fields
- Normalize labels
- Normalize dataset split values
- Detect duplicate record IDs
- Detect missing record IDs
- Detect missing text records
- Detect invalid labels
- Detect invalid dataset splits
- Export clean training data as CSV
- Export clean training data as JSONL
- Export rejected records with rejection reasons
- Generate a structured JSON pipeline summary
- Includes automated tests with `pytest`
- Uses a clean `src/` Python project structure

---

## Tech Stack

- Python
- Pandas
- Pydantic
- Typer
- Rich
- Pytest
- Ruff
- Mypy

---

## Project Structure

```text
ai-training-data-pipeline/
│
├── data/
│   ├── raw/
│   │   └── raw_training_data.csv
│   └── processed/
│
├── reports/
│
├── src/
│   └── ai_training_data_pipeline/
│       ├── __init__.py
│       ├── models.py
│       ├── loader.py
│       ├── cleaner.py
│       ├── validator.py
│       ├── exporter.py
│       ├── pipeline.py
│       └── cli.py
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore