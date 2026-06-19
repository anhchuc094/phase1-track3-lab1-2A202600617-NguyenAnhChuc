from __future__ import annotations
from pathlib import Path
import typer
from rich import print
from run_benchmark import main as run_benchmark

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/golden_test_set.json",
    out_dir: str = "outputs/golden_run",
    reflexion_attempts: int = 3,
) -> None:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise typer.BadParameter(
            f"Missing golden test set: {dataset_path}. Put the provided file there or pass --dataset PATH."
        )
    run_benchmark(
        dataset=str(dataset_path),
        out_dir=out_dir,
        reflexion_attempts=reflexion_attempts,
        min_mock_examples=1,
    )
    print(f"[green]Golden run complete[/green]: {out_dir}")

if __name__ == "__main__":
    app()
