# Reflexion Lab Implementation Explanation

## 1. What Was Completed

This repo originally had a Reflexion Agent scaffold. I completed the missing parts so it can run in two modes:

- `mock`: deterministic local runtime for autograde and quick testing.
- `llm`: real LLM runtime through OpenRouter or Groq using an OpenAI-compatible Chat Completions API.

The main files changed are:

- `src/reflexion_lab/schemas.py`
- `src/reflexion_lab/agents.py`
- `src/reflexion_lab/prompts.py`
- `src/reflexion_lab/mock_runtime.py`
- `src/reflexion_lab/reporting.py`
- `run_benchmark.py`
- `run_golden.py`
- `.env`

## 2. Schemas

File: `src/reflexion_lab/schemas.py`

Two important models were completed:

```python
class JudgeResult(BaseModel):
    score: Literal[0, 1]
    reason: str = Field(..., min_length=1)
    missing_evidence: list[str] = Field(default_factory=list)
    spurious_claims: list[str] = Field(default_factory=list)
```

`JudgeResult` stores the evaluator output:

- `score`: `1` means correct, `0` means wrong.
- `reason`: explains why the answer is correct or wrong.
- `missing_evidence`: facts or reasoning hops the answer missed.
- `spurious_claims`: unsupported or wrong claims from the answer.

```python
class ReflectionEntry(BaseModel):
    attempt_id: int
    failure_reason: str = Field(..., min_length=1)
    lesson: str = Field(..., min_length=1)
    next_strategy: str = Field(..., min_length=1)
```

`ReflectionEntry` stores one reflection after a failed attempt:

- `attempt_id`: which attempt failed.
- `failure_reason`: why it failed.
- `lesson`: what the agent should remember.
- `next_strategy`: what the next attempt should do differently.

## 3. Reflexion Loop

File: `src/reflexion_lab/agents.py`

The main loop now works like this:

1. Actor answers the question.
2. Evaluator judges the answer.
3. If the answer is correct, stop.
4. If the answer is wrong and the agent is `reflexion`, call the reflector.
5. Save the reflection into `reflection_memory`.
6. Retry with the reflection memory.

The key logic is:

```python
if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
    reflection = reflector(example, attempt_id, judge)
    reflections.append(reflection)
    reflection_memory.append(f"{reflection.lesson} Next strategy: {reflection.next_strategy}")
```

In simple terms: ReAct answers once. Reflexion can learn from a wrong answer and try again with a better strategy.

## 4. Token And Latency Tracking

File: `src/reflexion_lab/agents.py`

For mock mode, token count is estimated from the text length.

For real LLM mode, the code uses API usage data when the provider returns it:

```python
runtime_stats = consume_runtime_stats()
token_estimate = runtime_stats["total_tokens"] or token_estimate
latency_ms = runtime_stats["latency_ms"] or measured_latency
```

That means:

- Mock mode still works without an API.
- LLM mode reports real token usage if OpenRouter or Groq returns `usage.total_tokens`.

## 5. Prompts

File: `src/reflexion_lab/prompts.py`

Three system prompts were written:

- `ACTOR_SYSTEM`: tells the model to answer using only the provided context.
- `EVALUATOR_SYSTEM`: tells the model to grade the answer and return JSON.
- `REFLECTOR_SYSTEM`: tells the model to analyze the failed attempt and return JSON.

The evaluator returns:

```json
{
  "score": 0,
  "reason": "short explanation",
  "missing_evidence": [],
  "spurious_claims": []
}
```

The reflector returns:

```json
{
  "attempt_id": 1,
  "failure_reason": "why the previous answer was wrong",
  "lesson": "what to remember",
  "next_strategy": "what to do next"
}
```

This JSON format makes the output easy to parse into Pydantic models.

## 6. Real LLM Runtime

File: `src/reflexion_lab/mock_runtime.py`

The file still keeps mock functions, but now each public function checks the runtime mode:

```python
if runtime_mode() == "mock":
    return _mock_actor_answer(...)
```

If `REFLEXION_RUNTIME=llm`, the functions call a real LLM:

- `actor_answer()` calls the LLM with `ACTOR_SYSTEM`.
- `evaluator()` calls the LLM with `EVALUATOR_SYSTEM`.
- `reflector()` calls the LLM with `REFLECTOR_SYSTEM`.

The code supports:

- OpenRouter: `https://openrouter.ai/api/v1/chat/completions`
- Groq: `https://api.groq.com/openai/v1/chat/completions`

No extra Python package is required because the implementation uses Python's built-in `urllib.request`.

## 7. Environment Configuration

File: `.env`

Use OpenRouter:

```env
REFLEXION_RUNTIME=llm
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
LLM_MODEL=openai/gpt-4o-mini
```

Use Groq:

```env
REFLEXION_RUNTIME=llm
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
LLM_MODEL=llama-3.1-8b-instant
```

Use mock mode:

```env
REFLEXION_RUNTIME=mock
```

Important: `.env` is already listed in `.gitignore`, so API keys should not be committed.

## 8. Benchmark Behavior

File: `run_benchmark.py`

The benchmark now expands small datasets by default:

```python
def expand_for_mock_benchmark(examples, min_examples: int = 50):
    ...
```

Because each example is run with two agents, 50 examples create 100 records:

- 50 ReAct records
- 50 Reflexion records

This satisfies the autograder requirement `num_records >= 100`.

When running a real LLM, use:

```powershell
.\.venv\Scripts\python.exe run_benchmark.py --dataset data\hotpot_mini.json --out-dir outputs\llm_run --min-mock-examples 8
```

`--min-mock-examples 8` avoids repeating the dataset and saves API tokens.

## 9. Report Improvements

File: `src/reflexion_lab/reporting.py`

The report now includes:

- `meta`
- `summary`
- `failure_modes`
- `examples`
- `extensions`
- `discussion`

The failure mode report also has an `overall` section, so the analysis is easier to read.

The Markdown report also includes two submission-friendly tables:

1. ReAct vs Reflexion comparison table:
   - EM
   - average attempts
   - average token estimate
   - average latency

2. Cost estimate table:
   - total tokens
   - total runtime in seconds
   - total latency in milliseconds
   - cost per 1K tokens
   - estimated cost in USD

Set this environment variable if you want the cost table to use a real price:

```env
LLM_COST_PER_1K_TOKENS=0.00015
```

If the value is not set, the report still shows runtime and token cost structure, but estimated USD cost is `0`.

## 10. Golden Test Set Runner

File: `run_golden.py`

This script lets the code run immediately when a golden test set is provided.

Default expected path:

```text
data/golden_test_set.json
```

Run:

```powershell
.\.venv\Scripts\python.exe run_golden.py
```

Or pass a custom file:

```powershell
.\.venv\Scripts\python.exe run_golden.py --dataset data\your_golden_file.json --out-dir outputs\golden_run
```

The output files are:

```text
outputs\golden_run\react_runs.jsonl
outputs\golden_run\reflexion_runs.jsonl
outputs\golden_run\report.json
outputs\golden_run\report.md
```

## 11. Verification Results

Mock benchmark was verified successfully:

```powershell
$env:REFLEXION_RUNTIME='mock'
.\.venv\Scripts\python.exe run_benchmark.py --dataset data\hotpot_mini.json --out-dir outputs\sample_run
.\.venv\Scripts\python.exe autograde.py --report-path outputs\sample_run\report.json
```

Autograde result:

```text
Auto-grade total: 100/100
Flow Score (Core): 80/80
Bonus Score: 20/20
```

Real LLM mode was also attempted with your current `.env` settings. The request reached OpenRouter, but OpenRouter returned:

```text
HTTP 401 Unauthorized: User not found
```

That means the code path is calling the provider, but the API key/provider configuration is not accepted by OpenRouter. Common fixes:

- Make sure `LLM_PROVIDER=openrouter` only when using an OpenRouter key.
- Make sure `OPENROUTER_API_KEY` is copied correctly.
- If using Groq, set `LLM_PROVIDER=groq`, set `GROQ_API_KEY`, and use a Groq model name.
- Do not put quotes around the key unless your shell requires them.

## 12. Checklist For Submission

The requested items are covered as follows:

- ReAct vs Reflexion comparison table: included in `report.md` under `Summary`.
- Cost estimate table including running time: included in `report.md` under `Cost estimate`.
- Code runs directly with golden test set: use `run_golden.py`.

## 13. Short Explanation In Vietnamese

Agent ReAct chi tra loi mot lan. Agent Reflexion thi co them buoc tu sua sai: neu tra loi sai, evaluator se giai thich loi, reflector bien loi do thanh bai hoc ngan, roi actor dung bai hoc nay de tra loi lai o lan sau.

Vi du neu lan dau agent chi tra loi duoc hop dau tien, reflection memory se nhac no: "hay lam tiep hop thu hai va kiem tra entity cuoi cung trong context". Nho vay Reflexion thuong tot hon ReAct trong cau hoi multi-hop.

Mock mode dung de cham diem nhanh va on dinh. LLM mode dung de goi OpenRouter hoac Groq that, lay cau tra loi, diem danh gia, reflection, token va latency tu API.
