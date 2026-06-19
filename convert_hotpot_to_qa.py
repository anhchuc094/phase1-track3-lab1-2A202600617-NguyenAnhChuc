from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.schemas import QAExample

app = typer.Typer(add_completion=False)

def convert_item(item: dict) -> dict:
    context = []
    for chunk in item.get("context", []):
        if len(chunk) != 2:
            continue
        title, sentences = chunk
        text = " ".join(str(sentence).strip() for sentence in sentences).strip()
        context.append({"title": str(title), "text": text})
    return {
        "qid": str(item.get("_id", "")),
        "difficulty": item.get("level", "medium"),
        "question": item.get("question", ""),
        "gold_answer": item.get("answer", ""),
        "context": context,
    }

@app.command()
def main(
    input_path: str = "data/my_test_set.json",
    output_path: str = "data/my_test_set_qa.json",
    limit: int = 0,
) -> None:
    raw_path = Path(input_path)
    out_path = Path(output_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if limit > 0:
        raw = raw[:limit]
    converted = [convert_item(item) for item in raw]
    validated = [QAExample.model_validate(item).model_dump(mode="json") for item in converted]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[green]Converted[/green] {len(validated)} examples")
    print(f"[green]Saved[/green] {out_path}")

if __name__ == "__main__":
    app()
