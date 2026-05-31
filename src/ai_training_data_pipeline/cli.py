from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from ai_training_data_pipeline.pipeline import run_pipeline

app = typer.Typer(
    help="Clean, validate, and export AI training datasets.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def cli() -> None:
    """
    AI training data pipeline command-line interface.
    """


@app.command()
def run(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to the raw training data CSV file."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where processed data files should be saved.",
        ),
    ] = Path("data/processed"),
) -> None:
    """
    Run the AI training data pipeline.
    """
    try:
        outputs = run_pipeline(input_file=input_file, output_dir=output_dir)
        summary = outputs.summary

        console.print(
            Panel(
                f"""
AI TRAINING DATA PIPELINE SUMMARY
=================================

Raw records: {summary.raw_records}
Clean records: {summary.clean_records}
Rejected records: {summary.rejected_records}

Duplicate record IDs: {summary.duplicate_record_ids}
Missing text records: {summary.missing_text_records}
Invalid label records: {summary.invalid_label_records}
Invalid split records: {summary.invalid_split_records}

Train records: {summary.train_records}
Validation records: {summary.validation_records}
Test records: {summary.test_records}
""".strip(),
                title="AI Training Data Pipeline",
                expand=False,
            )
        )

        console.print(f"\nClean CSV saved to: [bold green]{outputs.clean_csv_path}[/bold green]")
        console.print(f"Clean JSONL saved to: [bold green]{outputs.clean_jsonl_path}[/bold green]")
        console.print(f"Rejected records saved to: [bold green]{outputs.rejected_csv_path}[/bold green]")
        console.print(f"Summary saved to: [bold green]{outputs.summary_json_path}[/bold green]")

    except Exception as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error


def main() -> None:
    app()


if __name__ == "__main__":
    main()