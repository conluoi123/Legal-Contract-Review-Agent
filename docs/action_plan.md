# 3-Month Action Plan — Legal Contract Review Agent

> Mục tiêu: AI Engineer Intern · Stack: LangGraph + RAG + FastAPI · Timeline: 12 tuần

---

## Tổng quan

| Phase                               | Tuần  | Mục tiêu                                           | Milestone                                |
| ----------------------------------- | ----- | -------------------------------------------------- | ---------------------------------------- |
| **Phase 1 — Eval & Observability**  | 1–3   | Có số đo thực, biết hệ thống hoạt động tốt đến đâu | F1, hallucination rate, ablation results |
| **Phase 2 — Retrieval Engineering** | 4–6   | Nâng chất lượng retrieval, track experiments       | Hybrid search + MLflow dashboard         |
| **Phase 3 — Production Pipeline**   | 7–10  | Không còn là hobby project                         | Docker + CI/CD + latency benchmark       |
| **Phase 4 — Ship & Evidence**       | 11–12 | Portfolio có thể defend trong interview            | Live demo + README + blog post           |

**Nguyên tắc xuyên suốt:** Mỗi improvement phải có số trước/sau. "Tôi cải thiện retrieval" không có giá trị. "Hybrid search tăng recall@5 từ 71% → 83%, logged trong MLflow run #14" mới là thứ interviewer nhớ.

---

## Phase 1 — Eval & Observability (Tuần 1–3)

### Tuần 1 — Tạo eval dataset & harness cơ bản

**Tại sao làm trước tiên:** Không có ground truth thì không biết mình đang ở đâu và cải thiện về đâu. Đây là foundation của mọi thứ sau.

**Task 1: Tạo 30 NDA test fixtures có ground truth**

Phân bổ:

- 10 safe contracts → expected: 0 HIGH risk, all clauses PRESENT
- 8 missing clause contracts → expected: specific missing clauses
- 8 risky contracts → expected: specific RiskItems với severity đúng
- 4 edge cases → expected behavior: graceful degradation hoặc HITL trigger

Với mỗi contract, tạo file `fixtures/nda_XX_expected.json`:

```json
{
  "contract_id": "nda_risk_liability_01",
  "expected_metadata": {
    "party_a": "Công ty ABC",
    "party_b": "Nguyễn Văn X",
    "effective_date": "2026-01-01"
  },
  "expected_risks": [
    {
      "clause_id": "2.3",
      "severity": "HIGH",
      "risk_type": "Liability Exclusion"
    }
  ],
  "expected_missing": [],
  "expected_hitl_trigger": false
}
```

**Task 2: Viết eval harness (pytest)**

```python
# tests/eval/eval_harness.py
def run_eval_on_dataset(fixtures_dir: str) -> EvalResults:
    results = []
    for fixture in glob(f"{fixtures_dir}/*.pdf"):
        expected = load_expected(fixture)
        actual = run_full_pipeline(fixture)
        results.append(compare(expected, actual))

    return EvalResults(
        precision=calculate_precision(results),
        recall=calculate_recall(results),
        f1=calculate_f1(results),
        hallucination_count=count_hallucinated_quotes(results),
        hitl_accuracy=calculate_hitl_accuracy(results)
    )
```

---

### Tuần 2 — DeepEval + Observability

**Task 1: Tích hợp DeepEval**

```python
# tests/eval/deepeval_suite.py
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric, AnswerRelevancyMetric

def test_faithfulness_across_dataset():
    metric = FaithfulnessMetric(threshold=0.95)
    for contract_path in glob("tests/fixtures/*.pdf"):
        contract_text = extract_full_text(contract_path)
        report = run_full_pipeline(contract_path)
        test_case = LLMTestCase(
            input=contract_text,
            actual_output=report.to_text_summary(),
            retrieval_context=[contract_text]
        )
        assert_test(test_case, [metric])

def test_zero_hallucinated_quotes():
    """Không có một quote nào được phép paraphrase"""
    for contract_path in glob("tests/fixtures/*.pdf"):
        contract_text = extract_full_text(contract_path)
        report = run_full_pipeline(contract_path)
        for risk in report.risks:
            assert risk.quote in contract_text, (
                f"HALLUCINATED QUOTE: '{risk.quote[:60]}...'\n"
                f"Not found verbatim in {contract_path}"
            )
```

**Task 2: Setup LangSmith hoặc Langfuse**

Mục tiêu: trace mọi LLM call với latency, token count, prompt version, response. Có dashboard thấy được.

```python
# Langfuse (free tier, self-hostable)
from langfuse.decorators import observe, langfuse_context

@observe()
async def detect_risk_for_clause(clause: Clause) -> RiskItem:
    langfuse_context.update_current_observation(
        metadata={"clause_id": clause.id, "clause_level": clause.level}
    )
    return await call_llm(build_risk_prompt(clause))
```

---

### Tuần 3 — Ablation Study + MLflow Setup

**Task 1: Ablation study prompt engineering**

Test 3 variants trên toàn bộ 30 contracts:

| Variant           | Mô tả                        | Expected                   |
| ----------------- | ---------------------------- | -------------------------- |
| `v1_zero_shot`    | Không có example             | Baseline                   |
| `v2_few_shot`     | 3 examples input→output      | +precision                 |
| `v3_cot_few_shot` | 3 examples với Thinking step | +recall, ít false positive |

```python
# scripts/ablation_study.py
PROMPT_VARIANTS = {
    "v1_zero_shot": ZERO_SHOT_PROMPT,
    "v2_few_shot": FEW_SHOT_PROMPT,
    "v3_cot": COT_FEW_SHOT_PROMPT,
}

for variant_name, prompt_template in PROMPT_VARIANTS.items():
    results = run_eval_with_prompt(prompt_template, fixtures_dir="tests/fixtures/")
    print(f"{variant_name}: F1={results.f1:.3f} | Precision={results.precision:.3f} | Recall={results.recall:.3f}")
```

**Task 2: Setup MLflow experiment tracking**

```python
# Mọi eval run đều được log vào MLflow
import mlflow

def run_and_log_experiment(
    prompt_version: str,
    similarity_threshold: float,
    use_hybrid_search: bool = False,
    use_reranker: bool = False,
):
    with mlflow.start_run(run_name=f"{prompt_version}_t{similarity_threshold}"):
        mlflow.log_params({
            "prompt_version": prompt_version,
            "similarity_threshold": similarity_threshold,
            "use_hybrid_search": use_hybrid_search,
            "use_reranker": use_reranker,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "claude-3-5-sonnet",
        })

        results = run_eval_on_dataset("tests/fixtures/")

        mlflow.log_metrics({
            "f1": results.f1,
            "precision": results.precision,
            "recall": results.recall,
            "hallucination_rate": results.hallucination_rate,
            "hitl_accuracy": results.hitl_accuracy,
            "p95_latency_s": results.p95_latency,
        })
```

**✓ Milestone 1:** Có số đo thực. Biết baseline F1, biết hallucination rate, biết prompt variant nào tốt nhất và tại sao.

---

## Phase 2 — Retrieval Engineering (Tuần 4–6)

### Tuần 4 — Hybrid Search

**Task 1: Triển khai BM25 + dense retrieval**

```python
# app/core/hybrid_search.py
from rank_bm25 import BM25Okapi

class HybridClauseChecker:
    """
    Kết hợp BM25 (exact match) + dense embedding (semantic).
    BM25 tốt cho: "điều khoản bất khả kháng", "điều 2.1.a", legal citations.
    Dense tốt cho: semantic similarity giữa clauses có nghĩa giống nhau.
    """

    def __init__(self, qdrant_client, bm25_weight=0.3, dense_weight=0.7):
        self.qdrant = qdrant_client
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # 1. Dense search
        query_embedding = embed_text(query)
        dense_results = self.qdrant.search(
            collection_name="nda_templates",
            query_vector=query_embedding,
            limit=top_k * 2  # Lấy nhiều hơn để RRF có đủ candidates
        )

        # 2. BM25 search trên template corpus
        bm25_results = self.bm25_index.get_top_n(
            query.split(), self.template_corpus, n=top_k * 2
        )

        # 3. Reciprocal Rank Fusion
        return self._rrf_merge(dense_results, bm25_results, top_k=top_k)

    def _rrf_merge(self, dense, bm25, top_k, k=60):
        """RRF score = Σ 1/(k + rank_i) cho mỗi result list"""
        scores = {}
        for rank, result in enumerate(dense):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
        for rank, result in enumerate(bm25):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**Task 2: Tune similarity threshold empirically**

```python
# scripts/tune_threshold.py
import numpy as np

def tune_threshold(fixtures_dir: str) -> float:
    best_f1, best_threshold = 0.0, 0.75

    for threshold in np.arange(0.50, 0.95, 0.05):
        results = run_eval_on_dataset(fixtures_dir, similarity_threshold=threshold)
        print(f"threshold={threshold:.2f} → F1={results.f1:.3f}")

        with mlflow.start_run(run_name=f"threshold_sweep_{threshold:.2f}"):
            mlflow.log_param("similarity_threshold", threshold)
            mlflow.log_metric("f1", results.f1)

        if results.f1 > best_f1:
            best_f1 = results.f1
            best_threshold = threshold

    print(f"\nOptimal threshold: {best_threshold} (F1={best_f1:.3f})")
    return best_threshold
```

---

### Tuần 5 — Reranker + Zero-Hallucination Enforcement

**Task 1: Cross-encoder reranker**

```python
# app/core/reranker.py
# Option A: Cohere Rerank (free tier — 1000 calls/month)
import cohere

class CohereReranker:
    def __init__(self):
        self.client = cohere.Client(api_key=COHERE_API_KEY)

    def rerank(self, query: str, candidates: List[str], top_n: int = 3) -> List[RankedResult]:
        response = self.client.rerank(
            query=query,
            documents=candidates,
            top_n=top_n,
            model="rerank-multilingual-v3.0"  # Hỗ trợ tiếng Việt
        )
        return response.results

# Option B: bge-reranker-base (local, không cần API key)
from sentence_transformers import CrossEncoder

class LocalReranker:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, candidates: List[str], top_n: int = 3):
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
```

**Task 2: instructor + Pydantic zero-hallucination**

```python
# app/models/report.py
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())

class RiskItem(BaseModel):
    clause_id: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    quote: str  # Phải nguyên văn — được enforce bằng validator bên dưới
    risk_type: str
    explanation: str
    suggestion: str
    beneficiary: str

    @validator("quote")
    def validate_verbatim_quote(cls, v, **kwargs):
        # original_text được inject vào context khi validate
        original = kwargs.get("values", {}).get("_original_text", "")
        if original and v not in original:
            raise ValueError(
                f"Quote must be verbatim from source. "
                f"Received: '{v[:60]}...' — not found in original clause text. "
                f"Copy-paste exactly."
                # instructor sẽ pass error này về LLM để tự sửa trong retry
            )
        return v

# Gọi với auto-retry
risk_item = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    response_model=RiskItem,
    messages=[{"role": "user", "content": build_risk_prompt(clause)}],
    max_retries=3  # Retry tự động khi validator fail
)
```

---

### Tuần 6 — Re-eval & Consolidate

**Task 1: Re-run full eval sau improvements**

```bash
# Chạy lại toàn bộ 30 contracts với stack mới
python scripts/run_full_eval.py \
  --prompt-version v3_cot \
  --threshold 0.75 \           # kết quả từ tuning tuần 4
  --hybrid-search true \
  --reranker cohere \
  --output results/eval_phase2_final.json
```

So sánh với baseline Phase 1:

| Metric              | Phase 1 Baseline | Phase 2 Final | Delta |
| ------------------- | ---------------- | ------------- | ----- |
| F1 (missing clause) | ~0.74            | ~0.85         | +11pp |
| Precision           | ~0.78            | ~0.87         | +9pp  |
| Recall@5            | ~0.71            | ~0.83         | +12pp |
| Hallucinated quotes | ~3%              | 0%            | -3pp  |
| P95 Latency         | ~35s             | ~22s          | -13s  |

**Task 2: MLflow comparison dashboard**

Sau ~20+ runs, dùng MLflow UI để visualize:

```bash
mlflow ui --port 5000
# → localhost:5000: parallel coordinates plot, metric comparison
```

Screenshot để đưa vào README — đây là evidence của engineering process.

**✓ Milestone 2:** Retrieval pipeline solid. Có số trước/sau rõ ràng. MLflow history thể hiện quá trình cải thiện có hệ thống.

---

## Phase 3 — Production Pipeline (Tuần 7–10)

### Tuần 7 — FastAPI Production-Grade + Docker

**Task 1: FastAPI production patterns**

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

logger = structlog.get_logger()

app = FastAPI(title="Legal Contract Review Agent", version="1.0.0")

# Middleware stack
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(RequestLoggingMiddleware)   # Log mọi request với request_id
app.add_middleware(RateLimitMiddleware, rate="10/minute")

@app.get("/health")
async def health_check():
    """Cần cho Docker healthcheck và uptime monitor"""
    return {
        "status": "ok",
        "qdrant": await check_qdrant_connection(),
        "postgres": await check_postgres_connection(),
        "llm_api": await check_llm_api_reachable(),
    }

# Structured logging — không dùng print()
logger.info(
    "risk_detected",
    contract_id=contract_id,
    clause_id=clause_id,
    severity=risk.severity,
    llm_latency_ms=round(latency * 1000),
    model="claude-3-5-sonnet",
    prompt_version="v3_cot",
)
```

**Task 2: Docker + docker-compose**

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://dev:dev@postgres:5432/contract_agent
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      qdrant: { condition: service_healthy }
      postgres: { condition: service_healthy }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s

  qdrant:
    image: qdrant/qdrant:v1.9.0 # Pin version
    ports: ["6333:6333"]
    volumes: ["./data/qdrant:/qdrant/storage"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: contract_agent
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    volumes: ["./data/postgres:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dev"]
```

---

### Tuần 8 — Reliability: Retry + Fallback + Async

**Task 1: Retry với exponential backoff + LLM fallback**

```python
# app/core/llm_client.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic, google.generativeai as genai

class ResilientLLMClient:
    """Primary: Claude 3.5 Sonnet. Fallback: Gemini Flash."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
        reraise=False,  # Không re-raise — trigger fallback
    )
    async def _call_claude(self, prompt: str) -> str:
        return await self.claude_client.complete(prompt)

    async def complete(self, prompt: str) -> str:
        try:
            return await self._call_claude(prompt)
        except Exception as e:
            logger.warning("claude_failed_using_fallback", error=str(e))
            # Fallback sang Gemini Flash
            return await self._call_gemini(prompt)

    async def _call_gemini(self, prompt: str) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await model.generate_content_async(prompt)
        return response.text
```

**Task 2: Async parallel risk detection**

```python
# app/agent/nodes/detect_risks.py
import asyncio

async def detect_risks_node(state: AgentState) -> AgentState:
    # Phân tích clauses song song thay vì tuần tự
    # 30 clauses × 2s = 60s tuần tự → ~6s parallel (với 10 concurrent)
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent LLM calls

    async def detect_with_semaphore(clause: Clause):
        async with semaphore:
            try:
                return await detect_risk_for_clause(clause)
            except Exception as e:
                state.errors.append(f"Risk detection failed: {clause.id}: {e}")
                return None  # Không crash toàn bộ pipeline

    tasks = [
        detect_with_semaphore(clause)
        for clause in state.clauses
        if clause.level >= 2  # Chỉ analyze Mục trở xuống
    ]

    results = await asyncio.gather(*tasks)
    state.risks = [r for r in results if r is not None]
    return state
```

---

### Tuần 9 — Testing Pyramid + CI/CD

**Task 1: Testing pyramid hoàn chỉnh**

```
Unit tests (mock LLM)          → Chạy mỗi commit, < 30 giây
Integration tests (real LLM)   → Chạy khi open PR
LLM eval (DeepEval)            → Chạy khi merge vào main
```

```python
# Unit test — KHÔNG gọi real LLM
@pytest.fixture
def mock_llm(mocker):
    return mocker.patch(
        "app.core.llm_client.ResilientLLMClient.complete",
        return_value='{"severity":"HIGH","quote":"...","risk_type":"Liability Exclusion",...}'
    )

def test_risk_detection_high_severity(mock_llm):
    clause = Clause(id="2.3", content="Bên A không chịu trách nhiệm trong mọi trường hợp...")
    risk = detect_risk(clause)
    assert risk.severity == "HIGH"
    assert risk.risk_type == "Liability Exclusion"
    mock_llm.assert_called_once()  # Verify đúng 1 LLM call
```

**Task 2: GitHub Actions CI/CD**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: "pip" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --tb=short
        env: { SKIP_LLM_CALLS: "true" }

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'pull_request'
    services:
      qdrant: { image: qdrant/qdrant:v1.9.0, ports: ["6333:6333"] }
      postgres:
        image: postgres:16-alpine
        env: { POSTGRES_PASSWORD: test }
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  llm-eval:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main' # Chỉ khi merge vào main
    steps:
      - run: pytest tests/eval/ -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

---

### Tuần 10 — Benchmark + Security

**Task 1: Latency benchmark script**

```python
# scripts/benchmark.py
import asyncio, time
from statistics import mean, stdev

async def benchmark_pipeline(pdf_paths: list[str], n_runs: int = 3):
    all_timings = []

    for pdf in pdf_paths:
        for _ in range(n_runs):
            timings = {}
            t0 = time.perf_counter()

            clauses = parse_document(pdf)
            timings["parsing"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            metadata = await extract_entities(clauses)
            timings["ner"] = time.perf_counter() - t1

            t2 = time.perf_counter()
            embeddings = batch_embed([c.content for c in clauses])
            timings["embedding"] = time.perf_counter() - t2

            t3 = time.perf_counter()
            risks = await detect_risks_parallel(clauses)
            timings["risk_detection"] = time.perf_counter() - t3

            timings["total"] = sum(timings.values())
            all_timings.append(timings)

    # Report
    for step in ["parsing", "ner", "embedding", "risk_detection", "total"]:
        values = [t[step] for t in all_timings]
        p95 = sorted(values)[int(0.95 * len(values))]
        print(f"{step:20s}: mean={mean(values):.2f}s  p95={p95:.2f}s  std={stdev(values):.2f}s")

    total_p95 = sorted([t["total"] for t in all_timings])[int(0.95 * len(all_timings))]
    assert total_p95 < 30, f"Performance regression: P95={total_p95:.2f}s > 30s target"

if __name__ == "__main__":
    asyncio.run(benchmark_pipeline(glob("tests/fixtures/*.pdf")))
```

**Task 2: Security basics**

```python
# File upload validation
async def validate_upload(file: UploadFile) -> None:
    ALLOWED_MIMES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    MAX_SIZE_MB = 20

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File vượt quá {MAX_SIZE_MB}MB")

    # Dùng python-magic để verify MIME thực sự (không trust header)
    import magic
    actual_mime = magic.from_buffer(content, mime=True)
    if actual_mime not in ALLOWED_MIMES:
        raise HTTPException(400, "Chỉ chấp nhận PDF và DOCX")

# Qdrant filter bắt buộc — tránh data leak giữa users
def search_clauses(query_vector, contract_id: str):
    return qdrant.search(
        collection_name="contract_clauses",
        query_vector=query_vector,
        query_filter=Filter(             # BẮT BUỘC — không query global
            must=[FieldCondition(
                key="contract_id",
                match=MatchValue(value=contract_id)
            )]
        ),
        limit=5
    )
```

**✓ Milestone 3:** Production-ready. CI/CD pass, Docker chạy được, retry hoạt động, latency P95 < 18s. Không còn là hobby project.

---

## Phase 4 — Ship & Evidence (Tuần 11–12)

### Tuần 11 — Deploy Live Demo + README

**Task 1: Deploy lên Railway/Render**

```bash
# Railway deployment
railway login
railway init
railway up

# Environment variables cần set:
# ANTHROPIC_API_KEY, OPENAI_API_KEY, DATABASE_URL, QDRANT_URL
```

Yêu cầu sau khi deploy:

- Link chạy được 24/7 (không free tier sleep sau 15 phút)
- Health check endpoint `/health` trả 200
- UptimeRobot monitor (free) để biết khi nào down
- Response time < 30s cho demo request thực tế

**Task 2: README chất lượng**

README phải có đủ các phần sau:

```markdown
# Legal Contract Review Agent

> AI Agent phân tích NDA tự động — phát hiện rủi ro, điều khoản thiếu,
> tương tác Human-in-the-loop khi cần.

## Live Demo

[link] | [demo GIF 30 giây]

## Eval Results (30-contract dataset)

| Metric                        | Score |
| ----------------------------- | ----- |
| F1 (missing clause detection) | 0.85  |
| Precision (risk detection)    | 0.87  |
| Hallucinated quotes           | 0%    |
| P95 latency (10-page NDA)     | 17.3s |

## Architecture

[diagram — Mermaid hoặc draw.io export]

## Key Technical Decisions

- Hierarchical chunking vs flat: [lý do]
- Hybrid BM25 + dense vs dense-only: [số đo]
- LangGraph vs custom loop: [lý do]
- instructor vs raw JSON: [lý do]

## Experiment Tracking

30+ runs logged in MLflow. Optimal config found:

- prompt_version: v3_cot (F1 +11pp vs zero-shot)
- similarity_threshold: 0.75 (F1-optimal từ sweep)
- hybrid_search: true (recall@5 +12pp)

## Quick Start

\`\`\`bash
docker-compose up
curl -X POST http://localhost:8000/api/v1/contracts/upload -F "file=@sample_nda.pdf"
\`\`\`
```

---

### Tuần 12 — Blog Post + Mock Interview Prep

**Task 1: 1 blog post kỹ thuật**

Chủ đề: **"Tại sao Hierarchical Chunking tốt hơn Flat Chunking cho Legal Documents"**

Cấu trúc:

- Problem: flat chunking mất context pháp lý như thế nào (ví dụ cụ thể)
- Solution: hierarchical chunking với parent_id và full_path
- Kết quả: so sánh F1 trước/sau (nếu có)
- Code snippet ngắn minh họa

Đăng lên: Medium hoặc Substack. Mục tiêu không phải viral — mục tiêu là exist và có link đặt trong CV.

**Task 2: Mock interview prep — 10 câu phải trả lời được ngay**

Với mỗi câu, phải có **con số thực** từ project:

1. "F1 score của hệ thống là bao nhiêu?" → "0.85 trên 30-contract test set"
2. "Tại sao dùng LangGraph?" → "Native interrupt() cho HITL, stateful checkpoint..."
3. "Hybrid search giúp ích gì?" → "Recall@5 tăng từ 71% lên 83%"
4. "Nếu LLM API down thì sao?" → "Fallback sang Gemini Flash sau 3 retry..."
5. "Latency của pipeline?" → "P95 = 17.3s cho 10-trang NDA, measured với benchmark script"
6. "Tại sao Hierarchical Chunking?" → "Flat chunking mất context pháp lý..."
7. "Quote hallucination rate?" → "0% — enforced bằng Pydantic validator + instructor retry"
8. "Testing strategy?" → "3 tiers: unit (mock LLM) / integration / LLM eval, CI/CD GitHub Actions"
9. "Cosine threshold 0.75 từ đâu ra?" → "Empirical sweep 0.5→0.95, F1-optimal trên validation set"
10. "Scale như thế nào?" → "async parallel, LangGraph thread isolation, Qdrant payload filter..."

**✓ Milestone 4:** Portfolio done. Apply được ngay. Mọi claim trên CV đều có evidence defend được trong interview.

---

## Acceptance Criteria cuối cùng

| Metric                        | Target       | Cách đo                            |
| ----------------------------- | ------------ | ---------------------------------- |
| F1 (missing clause detection) | ≥ 0.85       | Eval harness trên 30 contracts     |
| Precision (risk detection)    | ≥ 0.80       | Labeled dataset                    |
| Hallucinated quotes           | 0%           | Substring check toàn bộ quotes     |
| Hallucination rate (DeepEval) | < 5%         | HallucinationMetric                |
| P95 latency (10-trang NDA)    | < 20s        | Benchmark script, 10 PDFs × 3 runs |
| CI/CD                         | Pass         | GitHub Actions green               |
| Live demo                     | Uptime > 99% | UptimeRobot monitor                |
| MLflow experiments            | ≥ 20 runs    | Logged với params + metrics đầy đủ |

---

## CV Lines sau 3 tháng

```
Legal Contract Review Agent  |  Python · LangGraph · FastAPI · Qdrant · Claude API

• Built end-to-end NDA analysis agent using LangGraph HITL + hierarchical chunking;
  evaluated on 30-contract dataset achieving F1=0.85 for missing clause detection

• Implemented hybrid BM25 + dense retrieval with cross-encoder reranking;
  improved recall@5 by 12pp vs dense-only baseline, tracked via MLflow (30+ experiments)

• Enforced zero hallucination via Pydantic validators + instructor auto-retry;
  0% paraphrased quotes across full test set

• Deployed async FastAPI pipeline (Docker + GitHub Actions CI/CD);
  P95 latency 17s for 10-page NDA with parallel clause processing
```

---

## Apply timeline

**Không đợi tuần 12 mới apply.** Gửi CV từ tuần 8–9 khi Phase 3 đã xong — lúc đó bạn đã có Docker, CI/CD, eval metrics, và latency numbers. Tuần 11–12 là polish thêm, không phải prerequisite.

Học từ feedback thực tế sớm hơn sẽ tốt hơn là chờ "perfect".
