from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any
from dotenv import load_dotenv
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
_RUNTIME_STATS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0}

load_dotenv()

def runtime_mode() -> str:
    return os.getenv("REFLEXION_RUNTIME", "mock").strip().lower()

def reset_runtime_stats() -> None:
    for key in _RUNTIME_STATS:
        _RUNTIME_STATS[key] = 0

def consume_runtime_stats() -> dict[str, int]:
    stats = dict(_RUNTIME_STATS)
    reset_runtime_stats()
    return stats

def _provider_config() -> tuple[str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "openrouter":
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    elif provider == "groq":
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    else:
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENROUTER_API_KEY, GROQ_API_KEY, or LLM_API_KEY.")
    return base_url.rstrip("/"), api_key, model

def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in example.context)

def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM did not return a JSON object: {text}")
    return json.loads(text[start : end + 1])

def _chat(system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
    base_url, api_key, model = _provider_config()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if "json" in system_prompt.lower():
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": os.getenv("LLM_USER_AGENT", "reflexion-lab/1.0"),
    }
    if os.getenv("LLM_PROVIDER", "").strip().lower() == "openrouter":
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "reflexion-lab")
    start = time.perf_counter()
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds = float(retry_after)
                else:
                    match = re.search(r"try again in ([0-9.]+)s", detail, flags=re.IGNORECASE)
                    wait_seconds = float(match.group(1)) if match else 2.0
                time.sleep(wait_seconds + 0.25)
                continue
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < max_retries:
                time.sleep(2.0 + attempt)
                continue
            raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
    latency_ms = max(1, round((time.perf_counter() - start) * 1000))
    usage = data.get("usage", {})
    _RUNTIME_STATS["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    _RUNTIME_STATS["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    _RUNTIME_STATS["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    _RUNTIME_STATS["latency_ms"] += latency_ms
    return data["choices"][0]["message"]["content"].strip()

def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer

def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if runtime_mode() == "mock":
        return _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)
    user_prompt = f"""Question:
{example.question}

Context:
{_format_context(example)}

Attempt: {attempt_id}
Agent type: {agent_type}
Reflection memory:
{json.dumps(reflection_memory, ensure_ascii=False, indent=2)}

Return only the final answer."""
    return _chat(ACTOR_SYSTEM, user_prompt, temperature=float(os.getenv("LLM_ACTOR_TEMPERATURE", "0.2")))

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if runtime_mode() == "mock":
        return _mock_evaluator(example, answer)
    user_prompt = f"""Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}

Context:
{_format_context(example)}

Return the judgment JSON."""
    return JudgeResult.model_validate(_extract_json_object(_chat(EVALUATOR_SYSTEM, user_prompt)))

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if runtime_mode() == "mock":
        return _mock_reflector(example, attempt_id, judge)
    user_prompt = f"""Attempt id:
{attempt_id}

Question:
{example.question}

Context:
{_format_context(example)}

Evaluator feedback:
{judge.model_dump_json()}

Return the reflection JSON."""
    data = _extract_json_object(_chat(REFLECTOR_SYSTEM, user_prompt, temperature=float(os.getenv("LLM_REFLECTOR_TEMPERATURE", "0.1"))))
    data["attempt_id"] = attempt_id
    return ReflectionEntry.model_validate(data)
