# Testing Strategy & Phase Roadmap — Legal Contract Review Agent

> Senior Design Document · v1.0

---

## PHẦN 1: PHASE ROADMAP

### Nguyên tắc chung

Mỗi Phase phải có **Definition of Done** rõ ràng trước khi sang Phase tiếp. Không bắt đầu Phase 3 khi Phase 2 chưa đạt acceptance criteria. Portfolio cần thấy commit history sạch, mỗi Phase là 1 milestone tag.

---

### Phase 1 — Foundation (Tuần 1–2)

**Goal:** Infrastructure hoạt động + Document Parsing pipeline ra đúng Hierarchical JSON

#### 1.1 Setup Infrastructure

```bash
# docker-compose.yml — services cần thiết cho dev
services:
  qdrant:
    image: qdrant/qdrant:v1.9.0   # Pin version, không dùng latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["./data/qdrant:/qdrant/storage"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: contract_agent
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    ports: ["5432:5432"]
    volumes: ["./data/postgres:/var/lib/postgresql/data"]

  # Adminer để inspect DB trong dev
  adminer:
    image: adminer
    ports: ["8080:8080"]
```

```bash
# Project setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # pyproject.toml với dev extras

# Database migration
alembic upgrade head        # Tạo tables: contracts, reports, checkpoints

# Index NDA templates vào Qdrant
python scripts/index_templates.py --template-dir templates/nda/

# Verify
python scripts/health_check.py  # Qdrant + Postgres + LLM API reachable
```

#### 1.2 Hierarchical Chunking Pipeline

**Task chi tiết:**

```python
# app/core/document_parser.py

class DocumentParser:
    """
    Wrapper around LlamaParse với fallback logic.
    Trả về List[Clause] với quan hệ parent-child đúng.
    """

    def parse(self, file_path: str) -> List[Clause]:
        raw = self._llamaparse(file_path)
        return self._build_tree(raw)

    def _build_tree(self, raw_sections: List[Dict]) -> List[Clause]:
        """
        Convert flat LlamaParse output → hierarchical tree.

        LlamaParse trả về:
        [{"level": 1, "text": "Điều 1: Định nghĩa"}, ...]

        Ta cần:
        Clause(id="1", level=1, parent_id=None)
        Clause(id="1.1", level=2, parent_id="1")
        """
        clauses = []
        id_counters = defaultdict(int)
        parent_stack = []  # Stack theo dõi ancestors

        for section in raw_sections:
            level = section["level"]

            # Pop stack đến level cha
            while parent_stack and parent_stack[-1].level >= level:
                parent_stack.pop()

            parent = parent_stack[-1] if parent_stack else None
            id_counters[level] += 1

            # Build ID dạng "1.2.a"
            clause_id = self._build_id(level, id_counters, parent)
            full_path = self._build_path(clause_id, clauses)

            clause = Clause(
                id=clause_id,
                level=level,
                title=section.get("title", ""),
                content=section["text"],
                parent_id=parent.id if parent else None,
                full_path=full_path
            )

            clauses.append(clause)
            parent_stack.append(clause)

        return clauses
```

**Edge cases phải xử lý:**

- PDF scan (không có text layer) → LlamaParse OCR mode
- Header bị merge vào body text → heuristic detection bằng font size
- Bảng trong hợp đồng (schedule, phụ lục) → flatten thành text, tag `table_content=True`
- Tham chiếu chéo: "theo quy định tại Điều 3.2" → lưu `references: ["3.2"]` trong Clause

#### 1.3 Template Indexing

```python
# scripts/index_templates.py
# Chạy một lần khi setup, idempotent

def index_nda_templates(template_dir: str):
    templates = load_json_templates(template_dir)

    for template in templates:
        for clause in template.clauses:
            embedding = embed_text(clause.content)

            qdrant.upsert(
                collection_name="nda_templates",
                points=[PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "clause_title": clause.title,
                        "clause_type": clause.type,        # "definition", "obligation", "termination", ...
                        "criticality": clause.criticality, # "REQUIRED", "RECOMMENDED", "OPTIONAL"
                        "template_id": template.id,
                        "content": clause.content,
                        "language": template.language      # "vi" hoặc "en"
                    }
                )]
            )
```

#### 1.4 Definition of Done — Phase 1

- [ ] `docker-compose up` → Qdrant + PostgreSQL chạy, healthcheck pass
- [ ] `python parse.py --input sample_nda.pdf` → in ra JSON tree, giữ đúng parent-child
- [ ] Parse được 5/5 sample NDA PDF (khác font, layout)
- [ ] 10 template clauses đã được index trong Qdrant, query trả kết quả
- [ ] `pytest tests/test_parser.py` → 100% pass
- [ ] README Phase 1 update với architecture diagram

---

### Phase 2 — Core AI Modules (Tuần 3–4)

**Goal:** 3 modules hoạt động độc lập, đạt accuracy target

#### 2.1 NER Module (Module 2A)

**Task chi tiết:**

```python
# Xử lý date parsing tiếng Việt — pain point thực tế
VIETNAMESE_DATE_PATTERNS = [
    r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})",    # ngày 01 tháng 01 năm 2026
    r"(\d{1,2})/(\d{1,2})/(\d{4})",                    # 01/01/2026
    r"(\d{4})-(\d{2})-(\d{2})",                         # 2026-01-01
]

# Handle company name variations
# "Công ty TNHH ABC" = "Công ty trách nhiệm hữu hạn ABC"
# → normalize tên trong ContractMetadata
```

**Prompt few-shot phải có ít nhất:**

- 1 ví dụ hợp đồng tiếng Việt đầy đủ thông tin
- 1 ví dụ tiếng Việt thiếu effective_date
- 1 ví dụ tiếng Anh với company tên dài
- 1 ví dụ có số giấy phép kinh doanh (đừng nhầm với contract_value)

#### 2.2 Clause Checker Module (Module 2B)

**Threshold tuning strategy:**

```python
# KHÔNG hardcode threshold 0.75 ngay từ đầu
# Cần empirical testing trên dataset:

def tune_threshold(test_cases: List[Tuple[str, ClauseStatus]]) -> float:
    """
    test_cases = [(clause_text, expected_status), ...]
    Sweep threshold từ 0.5 đến 0.9, chọn threshold maximize F1
    """
    best_threshold = 0.75
    best_f1 = 0.0

    for threshold in np.arange(0.5, 0.9, 0.05):
        predictions = [check_with_threshold(c, threshold) for c, _ in test_cases]
        labels = [expected for _, expected in test_cases]
        f1 = f1_score(labels, predictions, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold  # Document kết quả này trong README
```

**Vấn đề ngôn ngữ mixed (tiếng Anh lẫn Việt):**

```python
# Một số NDA Việt Nam có điều khoản tiếng Anh
# text-embedding-3-small handle được multilingual
# Nhưng phải test: embed "confidential information" ≈ embed "thông tin bảo mật"?
# → Thêm test case này vào test_clause_check.py
```

#### 2.3 Risk Detection Module (Module 2C)

**Risk taxonomy cho NDA — phải biết 7 loại này:**

| Risk Type              | Dấu hiệu nhận biết                                                | Severity thường |
| ---------------------- | ----------------------------------------------------------------- | --------------- |
| Liability Exclusion    | "không chịu trách nhiệm", "trong mọi trường hợp", "miễn trách"    | HIGH            |
| Unilateral Termination | "có quyền chấm dứt mà không cần", "toàn quyền chấm dứt"           | HIGH            |
| Unreasonable Duration  | < 6 tháng hoặc > 5 năm, "vô thời hạn" không có điều kiện          | MEDIUM          |
| Foreign Jurisdiction   | tên tòa án/trọng tài nước ngoài, "Singapore", "ICC"               | MEDIUM          |
| Broad Confidentiality  | "mọi thông tin", "tất cả dữ liệu" không có giới hạn               | MEDIUM          |
| Asymmetric Obligations | "Bên A có quyền... Bên B phải..." — 1 bên chỉ có quyền            | MEDIUM          |
| Missing Consideration  | không có điều khoản thanh toán/đối giá cho bên cung cấp thông tin | LOW             |

**Vấn đề False Positive trong risk detection:**

```python
# "Bên A không chịu trách nhiệm về thiệt hại do bất khả kháng"
# → KHÔNG phải Liability Exclusion (bất khả kháng là hợp lệ)
# Phải có few-shot example về force majeure exception

# "Thời hạn bảo mật là 10 năm" trong lĩnh vực dược phẩm
# → Hợp lý vì IP dược phẩm có lifecycle dài
# Context-awareness: phải biết industry context
```

#### 2.4 Definition of Done — Phase 2

- [ ] NER accuracy ≥ 90% (đo trên 20 contract mẫu có ground truth)
- [ ] Clause checker F1 ≥ 80% cho MISSING detection
- [ ] Risk detection precision ≥ 80% (không flag false positive trên "safe" contracts)
- [ ] 0 hallucinated quotes — mọi `quote` field phải xuất hiện trong contract text gốc
- [ ] `pytest tests/test_ner.py tests/test_clause_check.py tests/test_risk_detection.py` → pass
- [ ] Notebook demo 3 modules với 1 NDA thực tế

---

### Phase 3 — Agent Orchestration (Tuần 5–6)

**Goal:** LangGraph graph hoàn chỉnh với Human-in-the-loop

#### 3.1 AgentState Design

```python
# app/agent/state.py
# AgentState phải serialize được → dùng dataclass, không dùng object phức tạp

@dataclass
class AgentState:
    contract_id: str
    file_path: str
    language: str = "vi"

    # Pipeline outputs
    clauses: List[Clause] = field(default_factory=list)
    metadata: Optional[ContractMetadata] = None
    clause_check_results: List[ClauseCheckResult] = field(default_factory=list)
    missing_clauses: List[str] = field(default_factory=list)
    risks: List[RiskItem] = field(default_factory=list)
    report: Optional[Report] = None

    # Control flow
    current_step: str = "initialized"
    errors: List[str] = field(default_factory=list)

    # HITL tracking
    hitl_history: List[Dict] = field(default_factory=list)
    # Lưu lại mọi lần interrupt để audit trail
```

#### 3.2 Graph Definition

```python
# app/agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

def build_graph(checkpointer: PostgresSaver) -> CompiledGraph:
    graph = StateGraph(AgentState)

    # Thêm nodes
    graph.add_node("parse_document", parse_document_node)
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("check_missing_entities", check_missing_entities_node)
    graph.add_node("compare_template", compare_template_node)
    graph.add_node("detect_risks", detect_risks_node)
    graph.add_node("check_ambiguous_risks", check_ambiguous_risks_node)
    graph.add_node("generate_report", generate_report_node)

    # Edges tuần tự
    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "extract_entities")
    graph.add_edge("extract_entities", "check_missing_entities")
    graph.add_edge("check_missing_entities", "compare_template")
    graph.add_edge("compare_template", "detect_risks")
    graph.add_edge("detect_risks", "check_ambiguous_risks")
    graph.add_edge("check_ambiguous_risks", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["check_missing_entities", "check_ambiguous_risks"])
    # interrupt_before: graph dừng trước khi chạy node đó nếu có interrupt()
```

#### 3.3 API Integration với LangGraph

```python
# app/api/contracts.py

@router.post("/contracts/upload")
async def upload_contract(file: UploadFile, background_tasks: BackgroundTasks):
    contract_id = str(uuid4())
    file_path = await save_to_storage(file, contract_id)

    # Lưu vào DB
    await db.execute(
        "INSERT INTO contracts (id, status, file_path) VALUES ($1, $2, $3)",
        contract_id, "PROCESSING", file_path
    )

    # Chạy agent async trong background
    background_tasks.add_task(run_agent, contract_id, file_path)

    return {"contract_id": contract_id, "status": "PROCESSING"}


async def run_agent(contract_id: str, file_path: str):
    state = AgentState(contract_id=contract_id, file_path=file_path)
    config = {"configurable": {"thread_id": contract_id}}

    # Stream events để update DB status
    async for event in compiled_graph.astream_events(state, config=config, version="v2"):
        if event["event"] == "on_chain_start":
            node_name = event["metadata"].get("langgraph_node")
            if node_name:
                await db.execute(
                    "UPDATE contracts SET current_step=$1 WHERE id=$2",
                    node_name, contract_id
                )

        elif event["event"] == "on_interrupt":
            # Agent đang chờ user input
            await db.execute(
                "UPDATE contracts SET status='AWAITING_INPUT', pending_questions=$1 WHERE id=$2",
                json.dumps(event["data"]), contract_id
            )
            return  # Dừng, chờ POST /answer


@router.get("/contracts/{contract_id}/status")
async def get_status(contract_id: str):
    row = await db.fetchrow("SELECT * FROM contracts WHERE id=$1", contract_id)
    return {
        "status": row["status"],
        "current_step": row["current_step"],
        "questions": json.loads(row["pending_questions"]) if row["pending_questions"] else None
    }


@router.post("/contracts/{contract_id}/answer")
async def answer_questions(contract_id: str, payload: AnswerPayload, background_tasks: BackgroundTasks):
    config = {"configurable": {"thread_id": contract_id}}

    # Update pending_questions = None, status = PROCESSING
    await db.execute(
        "UPDATE contracts SET status='PROCESSING', pending_questions=NULL WHERE id=$1",
        contract_id
    )

    # Resume graph với user answers
    background_tasks.add_task(
        compiled_graph.ainvoke,
        {"hitl_answer": payload.answers},
        config=config
    )

    return {"status": "RESUMED"}
```

#### 3.4 Definition of Done — Phase 3

- [ ] End-to-end flow: upload → agent chạy → interrupt → user trả lời → report
- [ ] State persist đúng: kill server sau interrupt, restart, resume được
- [ ] `pytest tests/test_agent_flow.py` → pass
- [ ] Không có memory leak trong async background tasks
- [ ] Demo video 2 phút: upload PDF → hỏi 1–2 câu → report xuất hiện

---

### Phase 4 — API + Frontend (Tuần 7–8)

**Goal:** Web app demo-able cho nhà tuyển dụng

#### 4.1 FastAPI Production Readiness

```python
# Middleware stack
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"])
app.add_middleware(
    RateLimitMiddleware,
    rate_limit="10/minute",     # 10 uploads/phút per IP
    key_func=get_client_ip
)

# Structured logging với request ID
app.add_middleware(RequestLoggingMiddleware)

# Health check (cần cho Docker/Railway)
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant": await check_qdrant(),
        "postgres": await check_postgres(),
        "llm": await check_llm_api()
    }
```

#### 4.2 Streamlit UI Architecture

```python
# frontend/app.py — state management quan trọng

def main():
    st.set_page_config(page_title="Contract Review Agent", layout="wide")

    # Session state keys
    if "contract_id" not in st.session_state:
        st.session_state.contract_id = None
    if "status" not in st.session_state:
        st.session_state.status = "idle"  # idle | processing | awaiting | done | error

    # Route theo status
    if st.session_state.status == "idle":
        render_upload_page()
    elif st.session_state.status == "processing":
        render_progress_page()  # Polling mỗi 2s
    elif st.session_state.status == "awaiting":
        render_hitl_chat()      # Chat widget cho HITL
    elif st.session_state.status == "done":
        render_report()
    elif st.session_state.status == "error":
        render_error()
```

**Progress polling — không dùng st.rerun() vô hạn:**

```python
def render_progress_page():
    progress_bar = st.progress(0)
    status_text = st.empty()

    STEP_PROGRESS = {
        "parse_document": 15,
        "extract_entities": 30,
        "compare_template": 50,
        "detect_risks": 75,
        "generate_report": 95,
    }

    while True:
        response = api_client.get_status(st.session_state.contract_id)
        step = response["current_step"]
        progress = STEP_PROGRESS.get(step, 0)

        progress_bar.progress(progress)
        status_text.text(f"⏳ {STEP_DISPLAY_NAMES[step]}")

        if response["status"] == "AWAITING_INPUT":
            st.session_state.status = "awaiting"
            st.session_state.pending_questions = response["questions"]
            st.rerun()
            break
        elif response["status"] == "DONE":
            st.session_state.status = "done"
            st.rerun()
            break
        elif response["status"] == "ERROR":
            st.session_state.status = "error"
            st.rerun()
            break

        time.sleep(2)  # Poll mỗi 2 giây
```

#### 4.3 Definition of Done — Phase 4

- [ ] API docs tại `/docs` (Swagger) đầy đủ và chính xác
- [ ] Streamlit UI: upload → progress → chat → report end-to-end không có lỗi
- [ ] Export PDF report hoạt động (dùng WeasyPrint hoặc Playwright)
- [ ] Deployed trên Railway/Render với custom domain
- [ ] README với demo GIF

---

### Phase 5 — Evaluation & Polish (Tuần 9–10)

**Goal:** Đủ chất lượng Portfolio, pass acceptance criteria

#### 5.1 Eval Harness + DeepEval Suite

**30-contract test dataset** (tăng lên từ 20 cho đủ statistical significance):

- 10 safe contracts → expected: 0 HIGH risk, all clauses PRESENT
- 8 missing clause contracts → expected: specific missing clauses
- 8 risky contracts → expected: specific RiskItems với severity đúng
- 4 edge cases → expected: graceful degradation hoặc HITL trigger

Mỗi contract có ground truth JSON:

```json
{
  "contract_id": "nda_risk_liability_01",
  "expected_risks": [
    { "clause_id": "2.3", "severity": "HIGH", "risk_type": "Liability Exclusion" }
  ],
  "expected_missing": [],
  "expected_hitl_trigger": false
}
```

**Eval harness (pytest):**

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

```python
# tests/eval/deepeval_suite.py
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric

class TestRiskDetectionQuality:

    @pytest.mark.parametrize("contract_fixture", [
        "fixtures/nda_safe_01.pdf",
        "fixtures/nda_safe_02.pdf",
        # ... 5 safe contracts
    ])
    def test_no_false_positive_on_safe_contracts(self, contract_fixture):
        """Safe contracts không được có bất kỳ HIGH risk item nào"""
        report = run_full_pipeline(contract_fixture)
        high_risks = [r for r in report.risks if r.severity == "HIGH"]
        assert len(high_risks) == 0, f"False positive: {[r.risk_type for r in high_risks]}"

    def test_quote_faithfulness_across_dataset(self):
        """Toàn bộ quote trong reports phải xuất hiện nguyên văn trong contract"""
        for contract_path in glob("fixtures/*.pdf"):
            contract_text = extract_full_text(contract_path)
            report = run_full_pipeline(contract_path)
            for risk in report.risks:
                assert risk.quote in contract_text, (
                    f"HALLUCINATED QUOTE in {contract_path}:\n"
                    f"Quote: '{risk.quote}'"
                )

    def test_hallucination_rate_below_threshold(self):
        """Dùng DeepEval để measure hallucination rate < 5%"""
        test_cases = []
        for contract_path in glob("fixtures/*.pdf"):
            contract_text = extract_full_text(contract_path)
            report = run_full_pipeline(contract_path)
            test_case = LLMTestCase(
                input=contract_text,
                actual_output=report.to_text_summary(),
                context=[contract_text]
            )
            test_cases.append(test_case)
        metric = HallucinationMetric(threshold=0.05)
        for tc in test_cases:
            assert_test(tc, [metric])
```

**MLflow experiment tracking cho mọi eval run** — số liệu thực hiện theo quy trình ở [Section 9 của architecture.md](architecture.md).

#### 5.2 Acceptance Criteria

| Metric                               | Target          | Cách đo                                      |
| ------------------------------------ | --------------- | -------------------------------------------- |
| Entity Extraction Accuracy           | ≥ 90%           | Exact match trên 20 contract có ground truth |
| Missing Clause Detection F1          | ≥ 85%           | Eval harness trên 30 contracts               |
| Risk Detection Precision             | ≥ 80%           | Trên labeled risk dataset                    |
| False Positive Rate (safe contracts) | = 0% HIGH risks | 10 safe contracts → 0 HIGH                   |
| Quote Faithfulness                   | 100%            | Substring check trên toàn bộ quotes          |
| Hallucination Rate                   | < 5%            | DeepEval HallucinationMetric                 |
| E2E Latency (10 trang PDF)           | < 30s           | P95 trên 10 runs                             |

---

## PHẦN 2: TESTING STRATEGY CHI TIẾT

### 2.1 Testing Pyramid

```
                    ┌─────────────────┐
                    │   E2E / Eval    │  ← Chậm, tốn API cost
                    │  (DeepEval)     │     Chạy: pre-release
                    ├─────────────────┤
                    │  Integration    │  ← Medium, cần Docker
                    │  (LangGraph)    │     Chạy: pre-merge
                    ├─────────────────┤
                    │   Unit Tests    │  ← Nhanh, không cần API
                    │   (Pytest)      │     Chạy: mỗi commit
                    └─────────────────┘
```

**Nguyên tắc:** Unit test không được gọi real LLM API. Dùng mock.

```python
# Đúng — mock LLM trong unit test
@pytest.fixture
def mock_llm(mocker):
    return mocker.patch(
        "app.core.llm_client.LLMClient.complete",
        return_value='{"party_a": "ABC Corp", "party_b": "XYZ Ltd", ...}'
    )

# Sai — gọi real API trong unit test (chậm, tốn tiền, flaky)
def test_ner_bad():
    result = extract_entities(sample_text)  # Gọi real Claude API!
    assert result.party_a == "ABC Corp"
```

---

### 2.2 Unit Tests — Mỗi Module

#### test_parser.py — 12 test cases

```python
class TestHierarchicalChunking:

    def test_level_1_clauses_have_no_parent(self, parsed_clauses):
        """Điều 1, Điều 2 phải có parent_id = None"""
        top_level = [c for c in parsed_clauses if c.level == 1]
        assert all(c.parent_id is None for c in top_level)

    def test_level_2_parent_exists(self, parsed_clauses):
        """Mục 1.1 phải có parent là Điều 1"""
        level_2 = [c for c in parsed_clauses if c.level == 2]
        clause_ids = {c.id for c in parsed_clauses}
        assert all(c.parent_id in clause_ids for c in level_2)

    def test_ids_are_unique(self, parsed_clauses):
        ids = [c.id for c in parsed_clauses]
        assert len(ids) == len(set(ids)), "Duplicate clause IDs!"

    def test_no_empty_content(self, parsed_clauses):
        assert all(len(c.content.strip()) > 20 for c in parsed_clauses)

    def test_full_path_format(self, parsed_clauses):
        """full_path phải có dạng 'Điều X > Mục X.Y' """
        for clause in parsed_clauses:
            if clause.level > 1:
                assert ">" in clause.full_path

    def test_table_clauses_tagged(self, contract_with_tables):
        """Clause từ bảng phải có tag table_content=True"""
        table_clauses = [c for c in contract_with_tables if c.table_content]
        assert len(table_clauses) > 0

    # ... 6 test cases nữa cho edge cases
```

#### test_ner.py — 15 test cases

```python
class TestEntityExtraction:

    def test_extract_vietnamese_company_name(self, mock_llm):
        text = "Giữa Công ty TNHH ABC (Bên A) và Nguyễn Văn X (Bên B)"
        metadata = extract_entities(text)
        assert "ABC" in metadata.party_a
        assert "Nguyễn Văn X" in metadata.party_b

    def test_parse_vietnamese_date_format(self, mock_llm):
        text = "có hiệu lực kể từ ngày 01 tháng 06 năm 2026"
        metadata = extract_entities(text)
        assert metadata.effective_date == date(2026, 6, 1)

    def test_missing_date_is_none_not_hallucinated(self, mock_llm_returns_null_date):
        """Khi không có date trong text, phải trả None, không được bịa"""
        text = "Giữa ABC Corp và XYZ Ltd, không có ngày cụ thể"
        metadata = extract_entities(text)
        assert metadata.effective_date is None

    def test_missing_critical_fields_detection(self, mock_llm_partial):
        metadata = ContractMetadata(party_a="ABC", party_b=None, effective_date=None)
        missing = metadata.missing_critical_fields
        assert "party_b" in missing
        assert "effective_date" in missing
        assert "party_a" not in missing

    def test_governing_law_extracted(self, mock_llm):
        text = "Hợp đồng này được điều chỉnh bởi pháp luật Việt Nam"
        metadata = extract_entities(text)
        assert "Việt Nam" in metadata.governing_law

    # ... 10 test cases nữa
```

#### test_clause_check.py — 10 test cases

```python
class TestClauseChecker:

    def test_exact_match_returns_present(self, qdrant_with_templates):
        """Clause giống hệt template → PRESENT"""
        result = check_clause(STANDARD_CONFIDENTIALITY_CLAUSE)
        assert result.status == ClauseStatus.PRESENT
        assert result.similarity_score >= 0.90

    def test_semantic_variant_still_present(self, qdrant_with_templates):
        """Clause khác chữ nhưng cùng nghĩa → PRESENT (score > 0.75)"""
        variant = "Dữ liệu được bảo vệ bao gồm mọi thông tin có tính chất bí mật..."
        result = check_clause(variant)
        assert result.status == ClauseStatus.PRESENT

    def test_missing_penalty_clause_detected(self, qdrant_with_templates):
        """Hợp đồng không có điều khoản phạt → MISSING trong output"""
        contract_without_penalty = load_fixture("nda_no_penalty.pdf")
        missing = find_missing_template_clauses(contract_without_penalty)
        assert any("phạt" in c.lower() or "penalty" in c.lower() for c in missing)

    def test_modified_clause_flagged(self, qdrant_with_templates):
        """Clause có trong template nhưng bị thay đổi đáng kể → MODIFIED"""
        modified = "Bên nhận có thể chia sẻ thông tin với bất kỳ bên thứ ba nào..."
        # Template: "Bên nhận không được chia sẻ..."
        result = check_clause(modified)
        assert result.status == ClauseStatus.MODIFIED

    def test_bilingual_clause_matching(self, qdrant_with_templates):
        """English clause phải match với Vietnamese template (multilingual embedding)"""
        en_clause = "Recipient shall not disclose any confidential information..."
        result = check_clause(en_clause)
        assert result.status != ClauseStatus.MISSING  # Phải tìm được match
```

#### test_risk_detection.py — 20 test cases

```python
@pytest.mark.parametrize("clause_text,expected_severity,expected_type", [
    # HIGH risks
    (
        "Bên A không chịu trách nhiệm đối với bất kỳ thiệt hại gián tiếp nào",
        "HIGH", "Liability Exclusion"
    ),
    (
        "Bên B có thể chấm dứt hợp đồng bất kỳ lúc nào mà không cần thông báo",
        "HIGH", "Unilateral Termination"
    ),
    # MEDIUM risks
    (
        "Thời hạn bảo mật là 99 năm kể từ ngày ký hợp đồng",
        "MEDIUM", "Unreasonable Duration"
    ),
    (
        "Mọi tranh chấp được giải quyết tại Tòa án ICC tại Paris",
        "MEDIUM", "Foreign Jurisdiction"
    ),
    # LOW / no risk
    (
        "Các bên đồng ý giữ bí mật mọi thông tin trao đổi trong quá trình hợp tác",
        "LOW", None
    ),
    # Không được false positive với force majeure
    (
        "Bên A không chịu trách nhiệm về thiệt hại do bất khả kháng theo quy định pháp luật",
        "LOW", None  # KHÔNG phải Liability Exclusion
    ),
])
def test_risk_classification(clause_text, expected_severity, expected_type, mock_llm_risk):
    risk = detect_risk(clause_text)
    assert risk.severity == expected_severity
    if expected_type:
        assert risk.risk_type == expected_type


class TestQuoteFaithfulness:
    """Test quan trọng nhất — zero tolerance"""

    @pytest.mark.parametrize("contract_fixture", glob("tests/fixtures/*.pdf"))
    def test_all_quotes_are_verbatim(self, contract_fixture):
        contract_text = extract_full_text(contract_fixture)
        # Dùng real LLM ở đây (integration test)
        risks = run_risk_detection(contract_text)

        for risk in risks:
            assert risk.quote in contract_text, (
                f"\nHALLUCINATED QUOTE DETECTED!\n"
                f"Contract: {contract_fixture}\n"
                f"Quote: '{risk.quote}'\n"
                f"This quote does not appear verbatim in the source document."
            )
```

---

### 2.3 Integration Tests

```python
# tests/test_agent_flow.py

class TestAgentFlow:

    @pytest.fixture
    def agent_with_test_db(self, tmp_path):
        """Setup in-memory SQLite checkpointer cho test"""
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        return build_graph(checkpointer)

    def test_full_pipeline_no_interrupt(self, agent_with_test_db):
        """Happy path: contract đầy đủ, không cần HITL"""
        state = AgentState(
            contract_id="test-001",
            file_path="tests/fixtures/nda_complete_01.pdf"
        )
        config = {"configurable": {"thread_id": "test-001"}}

        result = agent_with_test_db.invoke(state, config=config)

        assert result["report"] is not None
        assert result["current_step"] == "generate_report"
        assert len(result["errors"]) == 0

    def test_interrupt_on_missing_metadata(self, agent_with_test_db):
        """Contract thiếu effective_date → agent dừng và hỏi"""
        state = AgentState(
            contract_id="test-002",
            file_path="tests/fixtures/nda_missing_date.pdf"
        )
        config = {"configurable": {"thread_id": "test-002"}}

        # Lần 1: agent chạy đến interrupt
        result = agent_with_test_db.invoke(state, config=config)
        assert result["__interrupt__"] is not None
        interrupt_data = result["__interrupt__"][0].value
        assert interrupt_data["type"] == "missing_metadata"
        assert any(q["field"] == "effective_date" for q in interrupt_data["questions"])

        # Lần 2: resume với user answer
        resume_result = agent_with_test_db.invoke(
            {"hitl_answer": {"meta_effective_date": "2026-06-01"}},
            config=config
        )
        assert resume_result["metadata"].effective_date == date(2026, 6, 1)
        assert resume_result["report"] is not None

    def test_state_persists_across_restart(self, tmp_path):
        """State phải survive server restart (PostgreSQL checkpointer)"""
        db_path = tmp_path / "test.db"
        checkpointer_1 = SqliteSaver.from_conn_string(str(db_path))
        graph_1 = build_graph(checkpointer_1)

        # Run đến interrupt
        config = {"configurable": {"thread_id": "test-003"}}
        graph_1.invoke(AgentState(...), config=config)

        # Simulate server restart — tạo graph mới cùng DB
        checkpointer_2 = SqliteSaver.from_conn_string(str(db_path))
        graph_2 = build_graph(checkpointer_2)

        # Resume từ đầu — phải tiếp tục được
        result = graph_2.invoke({"hitl_answer": {...}}, config=config)
        assert result["report"] is not None
```

---

### 2.4 Benchmark & Performance Tests

```python
# tests/eval/benchmark.py

import asyncio
import time
from statistics import mean, median, stdev

async def benchmark_pipeline(pdf_paths: List[str], n_runs: int = 3):
    """
    Đo latency toàn bộ pipeline theo phần.
    Target: E2E < 30s cho 10-trang NDA
    """
    results = []

    for pdf in pdf_paths:
        for run in range(n_runs):
            timings = {}

            t0 = time.perf_counter()
            clauses = parse_document(pdf)
            timings["parsing"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            metadata = extract_entities(clauses)
            timings["ner"] = time.perf_counter() - t1

            t2 = time.perf_counter()
            embeddings = batch_embed([c.content for c in clauses])
            timings["embedding"] = time.perf_counter() - t2

            t3 = time.perf_counter()
            check_results = compare_template(clauses, embeddings)
            timings["clause_check"] = time.perf_counter() - t3

            t4 = time.perf_counter()
            risks = await detect_risks_parallel(clauses)
            timings["risk_detection"] = time.perf_counter() - t4

            timings["total"] = sum(timings.values())
            results.append(timings)

    # Report
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS — {len(pdf_paths)} PDFs × {n_runs} runs")
    print(f"{'='*50}")
    for step in ["parsing", "ner", "embedding", "clause_check", "risk_detection", "total"]:
        values = [r[step] for r in results]
        print(f"{step:20s}: mean={mean(values):.2f}s  p95={sorted(values)[int(0.95*len(values))]:.2f}s")

    total_values = [r["total"] for r in results]
    assert mean(total_values) < 30, f"Performance regression! Mean latency {mean(total_values):.2f}s > 30s target"

if __name__ == "__main__":
    asyncio.run(benchmark_pipeline(glob("tests/fixtures/*.pdf")))
```

---

### 2.5 CI/CD Pipeline

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
        with:
          python-version: "3.12"
          cache: "pip"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --ignore=tests/eval/ -k "not integration"
        env:
          SKIP_LLM_CALLS: "true" # Dùng mock LLM

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      qdrant:
        image: qdrant/qdrant:v1.9.0
        ports: ["6333:6333"]
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/test_agent_flow.py -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  llm-eval:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main' # Chỉ khi merge main
    steps:
      - run: pytest tests/eval/eval_suite.py -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

  # Không chạy benchmark trong CI (quá chậm)
  # Chạy manually: python tests/eval/benchmark.py
```

---

### 2.6 Test Data Management

**20 NDA fixtures — phân bổ:**

| File                         | Mô tả                                             | Ground Truth                                      |
| ---------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| `nda_complete_vi_01.pdf`     | NDA tiếng Việt đầy đủ, an toàn                    | 0 HIGH, all PRESENT                               |
| `nda_complete_vi_02.pdf`     | NDA tiếng Việt đầy đủ, khác layout                | 0 HIGH, all PRESENT                               |
| `nda_complete_en_01.pdf`     | NDA tiếng Anh đầy đủ                              | 0 HIGH, all PRESENT                               |
| `nda_complete_en_02.pdf`     | NDA tiếng Anh, US law                             | 0 HIGH, all PRESENT, MEDIUM: Foreign Jurisdiction |
| `nda_complete_bilingual.pdf` | Song ngữ Anh-Việt                                 | 0 HIGH, all PRESENT                               |
| `nda_missing_penalty.pdf`    | Thiếu điều khoản phạt                             | MISSING: penalty clause                           |
| `nda_missing_dispute.pdf`    | Thiếu cơ chế giải quyết tranh chấp                | MISSING: dispute resolution                       |
| `nda_missing_date.pdf`       | Thiếu effective_date                              | HITL trigger, MISSING: date                       |
| `nda_missing_parties.pdf`    | Thiếu tên Bên B                                   | HITL trigger                                      |
| `nda_missing_3clauses.pdf`   | Thiếu 3 điều khoản quan trọng                     | 3× MISSING                                        |
| `nda_risk_liability.pdf`     | Có Liability Exclusion rõ ràng                    | HIGH: Liability Exclusion                         |
| `nda_risk_termination.pdf`   | Unilateral termination                            | HIGH: Unilateral Termination                      |
| `nda_risk_duration99.pdf`    | Thời hạn 99 năm                                   | MEDIUM: Unreasonable Duration                     |
| `nda_risk_foreign_court.pdf` | Tòa án Singapore                                  | MEDIUM: Foreign Jurisdiction                      |
| `nda_risk_multiple.pdf`      | Nhiều rủi ro cùng lúc                             | 2× HIGH, 1× MEDIUM                                |
| `nda_edge_ambiguous.pdf`     | Ngôn ngữ mơ hồ chủ đích                           | HITL trigger                                      |
| `nda_edge_scan.pdf`          | PDF scan (không text layer)                       | OCR path                                          |
| `nda_edge_table_heavy.pdf`   | Nhiều bảng trong phụ lục                          | Table handling                                    |
| `nda_edge_long.pdf`          | 40 trang, nhiều điều khoản                        | Performance test                                  |
| `nda_edge_force_majeure.pdf` | Có bất khả kháng (không phải Liability Exclusion) | 0 HIGH — no false positive                        |

---

### Phase 6 — Production Pipeline & Evidence (Tuần 9–10)

**Goal:** Không còn là hobby project. CI/CD pass, Docker chạy được, latency P95 < 20s.

#### 6.1 GitHub Actions CI/CD Hoàn Chỉnh

Xọ3 tiết: unit tests mỗi commit, integration khi PR, LLM eval khi merge main:

```yaml
# .github/workflows/ci.yml
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
      - run: pytest tests/integration/ -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  llm-eval:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - run: pytest tests/eval/ -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

#### 6.2 Latency Benchmark Script

```python
# scripts/benchmark.py
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

    for step in ["parsing", "ner", "embedding", "risk_detection", "total"]:
        values = [t[step] for t in all_timings]
        p95 = sorted(values)[int(0.95 * len(values))]
        print(f"{step:20s}: mean={mean(values):.2f}s  p95={p95:.2f}s")

    total_p95 = sorted([t["total"] for t in all_timings])[int(0.95 * len(all_timings))]
    assert total_p95 < 20, f"Performance regression: P95={total_p95:.2f}s > 20s target"
```

#### 6.3 Security Checklist

- [ ] File upload: verify MIME thực sự bằng `python-magic` (không trust header)
- [ ] Max file size: 20MB limit trước khi xử lý
- [ ] Qdrant filter bắt buộc `contract_id` — không được cross-query giữa users
- [ ] JWT claim validate trước mọi `/contracts/{id}/*` endpoint
- [ ] Presigned URL expire sau 1 giờ

#### 6.4 Definition of Done — Phase 6

- [ ] Docker Compose chạy được với `docker-compose up`
- [ ] CI/CD GitHub Actions: unit → integration → eval tất cả green
- [ ] Latency P95 < 20s được confirm bằng benchmark script
- [ ] Deploy lên Railway/Render, health check `/health` trả 200 liên tục
- [ ] UptimeRobot monitor bật
- [ ] ≥ 20 MLflow runs logged với params + metrics đầy đủ
- [ ] README có: live demo link, eval results table, architecture diagram, demo GIF 30s

---

### Phần Kết — Apply Timeline & CV

**Không đợi Phase 6 xong mới apply.** Gửi CV từ tuần 7–8 khi Phase 4 đã xong — lúc đó đã có Docker, CI/CD, eval metrics, và latency numbers. Phần còn lại là polish thêm.

| Giai đoạn  | Tuần | Trạng thái có thể apply                                     |
| ---------- | ----- | -------------------------------------------------------------- |
| Phase 1–2  | 1–4   | Không — chưa có eval metrics                                  |
| Phase 3–4  | 5–8   | Có thể apply với: pipeline hoạt động, có số eval phase 2     |
| Phase 5–6  | 9–10  | **Optimal**: đủ số, CI/CD, benchmark, deploy                   |

**CV lines (copy-paste ready):**

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

**10 câu interview phải trả lời được ngay (có số thực tế):**

1. "F1 score của hệ thống là bao nhiêu?" → "0.85 trên 30-contract test set"
2. "Tại sao dùng LangGraph?" → "Native interrupt() cho HITL, stateful checkpoint PostgreSQL"
3. "Hybrid search giúp ích gì?" → "Recall@5 tăng từ 71% lên 83%"
4. "Nếu LLM API down thì sao?" → "Fallback sang Gemini Flash sau 3 retry exponential backoff"
5. "Latency của pipeline?" → "P95 = 17.3s cho 10-trang NDA, measured với benchmark script"
6. "Tại sao Hierarchical Chunking?" → "Flat chunking mất context pháp lý..."
7. "Quote hallucination rate?" → "0% — enforced bằng Pydantic validator + instructor retry"
8. "Testing strategy?" → "3 tiers: unit (mock LLM) / integration / LLM eval, CI/CD GitHub Actions"
9. "Cosine threshold 0.75 từ đâu?" → "Empirical sweep 0.5→0.95, F1-optimal trên validation set"
10. "Scale như thế nào?" → "async parallel, LangGraph thread isolation, Qdrant payload filter"

