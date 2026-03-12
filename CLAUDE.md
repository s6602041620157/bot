# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gemini PDF RAG Chatbot ("Askgiraffe") — a Thai-language Q&A chatbot for KMUTNB's Faculty of Technical Education. It answers questions from indexed PDF documents using RAG with a Streamlit UI.

**Stack:** Python 3.10, Streamlit, Google Gemini (`google-genai`), ChromaDB, PyMuPDF

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set GEMINI_API_KEY

# Index PDFs (place PDFs in data/pdfs/ first)
python scripts/index_pdfs.py --pdf-dir data/pdfs --reset

# Run the app
streamlit run app.py

# Tests
pytest -q                          # all tests
pytest tests/test_chunking.py -q   # single test file

# Benchmark & evaluation
python3 scripts/benchmark_chatbot_vs_gemini.py
python3 scripts/evaluate_ragas.py
```

## Architecture

### Query Flow (RAGPipeline.ask)

The pipeline in `rag/pipeline.py` orchestrates the full query flow with a multi-stage routing strategy:

1. **Retrieve** — embed query via `GeminiEmbedder`, search ChromaDB via `Retriever` (top_k chunks)
2. **Score gate** — filter chunks below `SIM_THRESHOLD` (`gates.apply_score_gate`)
3. **LLM relevance gate** — `LLMRelevanceGate` asks Gemini to judge if chunks are sufficient, returns `GateDecision` with confidence + kept_chunk_ids
4. **Route decision:**
   - `normal` — gate passes with high confidence → generate answer
   - `partial` — some high-scoring chunks exist but gate fails → generate with partial mode disclaimer
   - `fallback` — no PDF chunks pass → try QA fallback from `data/json/qa_prompt.json`
   - `refusal` — nothing relevant found → return canned refusal message
5. **Generate** — `GeminiAnswerGenerator` builds a structured prompt with context, chat history, mode instructions, and style policy, then calls Gemini
6. **Retry** — if generator returns a refusal, retry up to `GEN_RETRY_ON_REFUSAL` times

### Key Modules (rag/)

- `config.py` — `Settings` dataclass loaded from `.env` via `load_settings()`. All thresholds/toggles are env-configurable.
- `pipeline.py` — `RAGPipeline` is the main orchestrator. Thread-safe (uses `RLock`). Cached as `@st.cache_resource` in `app.py`.
- `gates.py` — Score filtering + LLM-based relevance judging. Returns `GateDecision`.
- `generator.py` — Prompt construction (`build_generation_prompt`) and Gemini generation. Supports 3 modes (normal/partial/fallback) and 3 style policies (auto/list/paragraph).
- `qa_fallback.py` — `QAFallbackRetriever` indexes Q&A pairs from a JSON file into a separate ChromaDB collection for fallback retrieval.
- `models.py` — Data classes: `DocumentChunk`, `RetrievedChunk`, `GateDecision`, `RAGAnswer`.
- `pdf_loader.py` — PDF text extraction via PyMuPDF.
- `chunker.py` — Text chunking with token counting.
- `embedder.py` — Wraps `google-genai` embedding API.
- `vector_store.py` — ChromaDB wrapper (upsert, query, reset).
- `retriever.py` — Combines embedder + vector_store for retrieval.

### Auto-heal

On startup, `pipeline.ensure_pdf_index_healthy()` checks if the index has enough documents. If not, it automatically re-indexes PDFs from `data/pdfs/`.

## Configuration

All settings are in `.env` (see `.env.example`). Key thresholds:
- `SIM_THRESHOLD` (0.55) — minimum cosine similarity for score gate
- `GATE_CONF_THRESHOLD` (0.65) — minimum LLM gate confidence
- `PARTIAL_MIN_SCORE` (0.68) / `PARTIAL_MIN_CHUNKS` (2) — partial answer eligibility
- `QA_SIM_THRESHOLD` (0.60) — fallback QA retrieval threshold

## Language

The chatbot is Thai-only. All prompts and responses are in Thai. The refusal message is defined in `rag/config.py:REFUSAL_MESSAGE`.
