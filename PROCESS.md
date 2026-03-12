# Askgiraffe — Process Document

This document covers both the **system internal process flow** (how queries are handled) and the **development/operations process** (how to set up, develop, test, and maintain the project).

---

## Part 1: System Process Flow

### 1.1 Indexing Process

The indexing pipeline transforms raw PDF files into searchable vector embeddings stored in ChromaDB.

```
data/pdfs/*.pdf
       |
       v
 +--------------+     +-------------+     +----------------+     +------------------+
 | pdf_loader   | --> | chunker     | --> | embedder       | --> | vector_store     |
 | (PyMuPDF)    |     | (900 chars, |     | (Gemini        |     | (ChromaDB,       |
 | extract text |     |  150 overlap)|    |  embedding-001)|     |  cosine space)   |
 | per page     |     | split text  |     | batch embed    |     | upsert chunks    |
 +--------------+     +-------------+     +----------------+     +------------------+
```

**Step-by-step:**

1. **List PDFs** — `pdf_loader.list_pdf_files()` scans `data/pdfs/` for `.pdf` files
2. **Extract pages** — `pdf_loader.extract_pdf_pages()` uses PyMuPDF (`fitz`) to extract text from each page, normalizing whitespace
3. **Chunk text** — `chunker.build_document_chunks()` splits page text into chunks of 900 characters with 150-character overlap, attempting word-boundary splits. Each chunk gets an ID like `filename.pdf:p1:c1`
4. **Embed** — `embedder.GeminiEmbedder.embed_texts()` sends chunks to Gemini's `gemini-embedding-001` model in batches of 32
5. **Store** — `vector_store.ChromaVectorStore.upsert_chunks()` stores chunk text, embedding vectors, and metadata (file_name, page, chunk_id, source_path, token_count) in ChromaDB with cosine distance

**QA Fallback Indexing:**

A separate collection (`qa_prompt_chunks`) is built from `data/json/qa_prompt.json`:

1. `qa_fallback.load_qa_entries()` parses the JSON array of `{id, question, answer, category, source_document}` objects
2. Each entry becomes a `DocumentChunk` with combined text: `"คำถาม: ... หมวดหมู่: ... คำตอบ: ..."`
3. `QAFallbackRetriever.ensure_index()` embeds and upserts into its own ChromaDB collection
4. If the collection count already matches the entry count, re-indexing is skipped

---

### 1.2 Query Process (RAGPipeline.ask)

The `ask()` method in `rag/pipeline.py` orchestrates a multi-stage routing strategy. The entire method is thread-safe via `RLock`.

```
User Query
    |
    v
+------------------+
| 1. RETRIEVE      |  Embed query, search ChromaDB (top_k=12)
+------------------+
    |
    v
+------------------+
| 2. SCORE GATE    |  Filter chunks below SIM_THRESHOLD (0.55)
+------------------+
    |
    +--- scored chunks exist?
    |       |
    |       v YES
    |   +------------------+
    |   | 3. LLM GATE     |  Gemini judges if chunks answer the query
    |   +------------------+  Returns: is_relevant, confidence, kept_chunk_ids
    |       |
    |       +--- is_relevant AND confidence >= 0.65?
    |       |       |
    |       |       v YES --> NORMAL route
    |       |       Generate answer with selected chunks
    |       |       If answer is a refusal AND partial eligible:
    |       |           Switch to PARTIAL route
    |       |
    |       +--- NO, but partial eligible?
    |               |
    |               v YES --> PARTIAL route
    |               Generate with high-scoring chunks (score >= 0.68, min 2)
    |
    +--- NO scored chunks (or partial not eligible)
            |
            v
    +------------------+
    | 4. QA FALLBACK   |  Search qa_prompt_chunks (top_k=5, threshold=0.60)
    +------------------+
        |
        +--- qa chunks found? --> FALLBACK route, generate answer
        |
        +--- nothing found --> REFUSAL route, return canned message
```

**Route details:**

| Route | Condition | Behavior |
|-------|-----------|----------|
| **normal** | Score gate passes + LLM gate relevant + confidence >= 0.65 | Generate answer from gate-selected chunks (up to 5) |
| **partial** | Some chunks score >= 0.68 (min 2) but gate fails or normal refused | Generate with disclaimer structure (ข้อมูลที่พบ / คำตอบ / คำถามแนะนำต่อ) |
| **fallback** | No PDF chunks pass, but QA JSON has a match (threshold 0.60) | Generate from QA fallback chunks with partial-style structure |
| **refusal** | Nothing relevant found anywhere | Return: "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่ให้มา กรุณาระบุคำถามให้เฉพาะเจาะจงขึ้น" |

**Retry logic:** If the generator returns a refusal, it retries up to `GEN_RETRY_ON_REFUSAL` (default 1) times before accepting the refusal.

**Answer generation (`generator.py`):**

The `build_generation_prompt()` function constructs a structured prompt with:
- System instructions (Thai-only, context-grounded, no external info)
- Mode instruction (normal vs partial/fallback)
- Style policy (auto/list/paragraph) — auto mode chooses format based on question type
- Chat history (last `MEMORY_TURNS * 2` messages)
- Context chunks with chunk_id, file, page references
- The user question

---

### 1.3 Auto-heal Process

On startup, `pipeline.ensure_pdf_index_healthy()` is called from `app.py` to ensure the index is populated.

```
App startup
    |
    v
auto_heal_index enabled? (default: True)
    |
    +--- NO --> skip, return current status
    |
    +--- YES
         |
         v
    Already checked this session?
         |
         +--- YES --> return cached status
         |
         +--- NO
              |
              v
         chunk_count > 0 AND document_count >= auto_heal_min_docs (1)?
              |
              +--- YES --> healthy, skip re-index
              |
              +--- NO --> call index_pdfs(reset=True) to rebuild from data/pdfs/
```

The result is cached so auto-heal runs at most once per pipeline instance lifetime (unless `force=True`).

---

### 1.4 Benchmark Process

`scripts/benchmark_chatbot_vs_gemini.py` compares the RAG chatbot against direct Gemini on the same dataset.

```
qa_prompt.json dataset
    |
    +---> RAGPipeline.ask(question) --> chatbot_answer
    |
    +---> DirectGeminiClient.answer_question(question, context) --> gemini_answer
    |
    v
GeminiJudge evaluates both answers against reference_answer
    |
    v
Output: reports/benchmark_results.csv + reports/benchmark_summary.json
```

**Key options:**
- `--gemini-context-mode` — `none` (no context), `qa_fallback` (QA chunks), or `pdf` (PDF chunks, default)
- `--gemini-context-top-k` — number of context chunks for direct Gemini (default 3)
- `--judge-model` — model used for judging (defaults to GEN_MODEL)
- `--limit N` — run only first N records for debugging

**Judge evaluation:** Uses a Gemini-as-judge approach — the judge prompt asks for `{is_correct, match_score, reason}` JSON. Scores are binary (correct/incorrect) with a match_score 0-1 for granularity. Results include breakdowns by `source_document` and `category`.

---

### 1.5 Evaluation Process (Ragas)

`scripts/evaluate_ragas.py` measures answer quality using the Ragas framework with four metrics.

```
qa_prompt.json dataset
    |
    v
RAGPipeline.ask(question) --> chatbot_answer + retrieved_contexts
    |
    v
Ragas evaluate() with LangChain-wrapped Gemini
    |
    v
Per-question scores:
  - faithfulness      (is the answer grounded in context?)
  - answer_relevancy  (does the answer address the question?)
  - context_precision (are retrieved chunks relevant?)
  - context_recall    (do retrieved chunks cover the reference answer?)
    |
    v
Output: reports/ragas_results.csv + reports/ragas_summary.json
```

**Key options:**
- `--ragas-llm-model` — model for Ragas LLM metrics (defaults to GEN_MODEL)
- `--ragas-embed-model` — model for Ragas embedding metrics (defaults to EMBED_MODEL)
- `--limit N` — run only first N records for debugging

---

## Part 2: Development & Operations Process

### 2.1 Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (required)
```

### 2.2 Data Preparation

**PDF documents:**
- Place PDF files in `data/pdfs/`
- All PDFs in this directory will be indexed automatically

**QA fallback JSON** (`data/json/qa_prompt.json`):
- JSON array of objects with fields: `id`, `question`, `answer`
- Optional fields: `source_document`, `category`
- Used as fallback when PDF chunks don't match, and as the evaluation dataset

### 2.3 Indexing

**Manual indexing:**
```bash
# Index all PDFs (additive)
python scripts/index_pdfs.py

# Index from a specific directory with reset
python scripts/index_pdfs.py --pdf-dir data/pdfs --reset
```

**Automatic indexing (auto-heal):**
- On app startup, `ensure_pdf_index_healthy()` checks if the index has documents
- If the index is empty or below `AUTO_HEAL_MIN_DOCS`, it re-indexes from `data/pdfs/` with `reset=True`
- Can be disabled with `AUTO_HEAL_INDEX=false` in `.env`

### 2.4 Running the App

```bash
streamlit run app.py
```

The Streamlit UI provides:
- Chat interface with Thai-language Q&A
- Sidebar with index health stats, document count, and manual re-index button
- Chat history (stored in `st.session_state`, cleared via sidebar button)

### 2.5 Testing

```bash
# Run all tests
pytest -q

# Run a specific test file
pytest tests/test_chunking.py -q
```

**Test files:**

| File | Coverage |
|------|----------|
| `test_chunking.py` | Text chunking logic and document chunk building |
| `test_relevance_gate.py` | Score gate filtering and LLM gate JSON parsing |
| `test_qa_fallback.py` | QA JSON loading and fallback retrieval |
| `test_generator_prompt.py` | Prompt construction for different modes and styles |
| `test_rag_flow.py` | End-to-end RAG pipeline routing (normal/partial/fallback/refusal) |
| `test_llm_config_passthrough.py` | Temperature/top_p config propagation |
| `test_eval_metrics.py` | Benchmark metric computation (binary metrics, breakdowns) |
| `test_benchmark_io.py` | Benchmark dataset loading, CSV/JSON output |
| `test_ragas_eval.py` | Ragas evaluation helpers (sample building, score merging) |
| `test_direct_gemini_prompt.py` | Direct Gemini client prompt construction |

### 2.6 Benchmarking & Evaluation

**Benchmark (chatbot vs direct Gemini):**
```bash
python3 scripts/benchmark_chatbot_vs_gemini.py

# With options
python3 scripts/benchmark_chatbot_vs_gemini.py \
  --dataset data/json/qa_prompt.json \
  --gemini-context-mode pdf \
  --limit 10
```

Output:
- `reports/benchmark_results.csv` — per-question results
- `reports/benchmark_summary.json` — aggregate metrics with breakdowns

**Ragas evaluation:**
```bash
python3 scripts/evaluate_ragas.py

# With options
python3 scripts/evaluate_ragas.py \
  --dataset data/json/qa_prompt.json \
  --limit 10
```

Output:
- `reports/ragas_results.csv` — per-question metric scores
- `reports/ragas_summary.json` — mean metrics across all questions

---

## Configuration Reference

All settings are loaded from `.env` via `rag/config.py:load_settings()`. Every setting has a default.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GEN_MODEL` | `gemini-2.5-flash` | Generation model |
| `EMBED_MODEL` | `gemini-embedding-001` | Embedding model |
| `PDF_DIR` | `data/pdfs` | PDF source directory |
| `CHROMA_DIR` | `storage/chroma` | ChromaDB persistence directory |
| `TOP_K` | `12` | Number of chunks to retrieve |
| `SIM_THRESHOLD` | `0.55` | Minimum cosine similarity for score gate |
| `GATE_CONF_THRESHOLD` | `0.65` | Minimum LLM gate confidence |
| `CHROMA_COLLECTION` | `pdf_rag_chunks` | ChromaDB collection name |
| `QA_JSON_PATH` | `data/json/qa_prompt.json` | QA fallback JSON path |
| `QA_COLLECTION` | `qa_prompt_chunks` | QA fallback collection name |
| `QA_TOP_K` | `5` | QA fallback retrieval count |
| `QA_SIM_THRESHOLD` | `0.60` | QA fallback similarity threshold |
| `PARTIAL_ENABLED` | `true` | Enable partial answer route |
| `PARTIAL_MIN_SCORE` | `0.68` | Minimum score for partial chunks |
| `PARTIAL_MIN_CHUNKS` | `2` | Minimum chunk count for partial answer |
| `ANSWER_STYLE_POLICY` | `auto` | Answer format: `auto`, `list`, or `paragraph` |
| `LOG_FILE` | `logs/app.log` | Log file path |
| `MEMORY_TURNS` | `6` | Chat history turns to include in prompt |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `LLM_TOP_P` | `1.0` | Generation top_p |
| `GEN_RETRY_ON_REFUSAL` | `1` | Max retries if generator returns refusal |
| `AUTO_HEAL_INDEX` | `true` | Enable auto-heal on startup |
| `AUTO_HEAL_MIN_DOCS` | `1` | Minimum indexed documents to be considered healthy |

---

## Project Structure

```
bot/
├── app.py                          # Streamlit UI entry point
├── rag/
│   ├── config.py                   # Settings dataclass, .env loading
│   ├── pipeline.py                 # RAGPipeline orchestrator
│   ├── gates.py                    # Score gate + LLM relevance gate
│   ├── generator.py                # Prompt building + Gemini generation
│   ├── qa_fallback.py              # QA JSON fallback retriever
│   ├── models.py                   # DocumentChunk, RetrievedChunk, GateDecision, RAGAnswer
│   ├── pdf_loader.py               # PDF text extraction (PyMuPDF)
│   ├── chunker.py                  # Text chunking with overlap
│   ├── embedder.py                 # Gemini embedding wrapper
│   ├── vector_store.py             # ChromaDB wrapper
│   ├── retriever.py                # Embedder + vector store retrieval
│   └── direct_gemini.py            # Direct Gemini client (benchmark baseline)
├── scripts/
│   ├── index_pdfs.py               # Manual PDF indexing CLI
│   ├── benchmark_chatbot_vs_gemini.py  # Chatbot vs Gemini benchmark
│   └── evaluate_ragas.py           # Ragas metrics evaluation
├── tests/                          # pytest test suite
├── data/
│   ├── pdfs/                       # Source PDF documents
│   └── json/qa_prompt.json         # QA fallback + evaluation dataset
├── storage/chroma/                 # ChromaDB persistence
├── reports/                        # Benchmark and evaluation outputs
├── logs/app.log                    # Application logs
└── .env                            # Environment configuration
```
