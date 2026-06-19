from __future__ import annotations
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
        overall[record.failure_mode] += 1
    breakdown = {agent: dict(counter) for agent, counter in grouped.items()}
    breakdown["overall"] = dict(overall)
    return breakdown

def cost_estimates(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    cost_per_1k_tokens = float(os.getenv("LLM_COST_PER_1K_TOKENS", "0") or 0)
    for record in records:
        grouped[record.agent_type].append(record)
    costs: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        total_tokens = sum(record.token_estimate for record in rows)
        total_latency_ms = sum(record.latency_ms for record in rows)
        costs[agent_type] = {
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "total_runtime_seconds": round(total_latency_ms / 1000, 3),
            "cost_per_1k_tokens_usd": cost_per_1k_tokens,
            "estimated_cost_usd": round((total_tokens / 1000) * cost_per_1k_tokens, 6),
        }
    return costs

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    summary = summarize(records)
    summary["cost_estimates"] = cost_estimates(records)
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summary, failure_modes=failure_breakdown(records), examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion="Reflexion improves the mock benchmark because it turns evaluator feedback into a short memory item before the next attempt. This is most useful for incomplete multi-hop answers, entity drift, and wrong final answers: the Actor is reminded to finish the second hop and verify the final entity against the context. The tradeoff is extra attempts, more token use, and additional latency. In a real LLM setting, the main remaining risk is evaluator quality: weak feedback can make the reflection vague or push the next attempt toward overfitting.")

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    costs = s.get("cost_estimates", {})
    react_cost = costs.get("react", {})
    reflexion_cost = costs.get("reflexion", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Cost estimate
| Metric | ReAct | Reflexion |
|---|---:|---:|
| Total tokens | {react_cost.get('total_tokens', 0)} | {reflexion_cost.get('total_tokens', 0)} |
| Total runtime (seconds) | {react_cost.get('total_runtime_seconds', 0)} | {reflexion_cost.get('total_runtime_seconds', 0)} |
| Total latency (ms) | {react_cost.get('total_latency_ms', 0)} | {reflexion_cost.get('total_latency_ms', 0)} |
| Cost per 1K tokens (USD) | {react_cost.get('cost_per_1k_tokens_usd', 0)} | {reflexion_cost.get('cost_per_1k_tokens_usd', 0)} |
| Estimated cost (USD) | {react_cost.get('estimated_cost_usd', 0)} | {reflexion_cost.get('estimated_cost_usd', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
