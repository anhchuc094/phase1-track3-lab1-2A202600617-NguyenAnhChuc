from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, consume_runtime_stats, evaluator, reset_runtime_stats, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

def _estimate_tokens(*parts: str) -> int:
    text = " ".join(part for part in parts if part)
    return max(1, round(len(text.split()) * 1.3))

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            reset_runtime_stats()
            start = perf_counter()
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            reflection = None
            final_answer = answer
            final_score = judge.score

            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(f"{reflection.lesson} Next strategy: {reflection.next_strategy}")

            token_estimate = _estimate_tokens(
                example.question,
                " ".join(chunk.text for chunk in example.context),
                " ".join(reflection_memory),
                answer,
                judge.reason,
                reflection.model_dump_json() if reflection else "",
            )
            runtime_stats = consume_runtime_stats()
            token_estimate = runtime_stats["total_tokens"] or token_estimate
            latency_ms = runtime_stats["latency_ms"] or max(1, round((perf_counter() - start) * 1000))
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            traces.append(trace)
            if judge.score == 1:
                break
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
