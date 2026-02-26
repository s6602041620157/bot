# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Thai-language RAG (Retrieval-Augmented Generation) chatbot built with Streamlit for answering questions about academic programs at the Faculty of Industrial Education, KMUTNB. The system uses Google's Gemini models for both generation and embeddings, with Chroma as the vector database.

## Common Development Commands

### Setup and Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Running the Application
```bash
streamlit run app.py
```

### Document Indexing
Index all PDF documents (required before first use):
```bash
python scripts/index_pdfs.py --pdf-dir data/pdfs --reset
```

### Testing
Run all tests:
```bash
pytest -q
```

### Benchmarking and Evaluation
Compare chatbot vs direct Gemini responses:
```bash
python3 scripts/benchmark_chatbot_vs_gemini.py
```

Evaluate with Ragas metrics:
```bash
python3 scripts/evaluate_ragas.py
```

## Architecture Overview

### Core Components (`rag/` module)
- **`pipeline.py`** - Main RAGPipeline class orchestrating the entire flow
- **`config.py`** - Settings management and environment variable handling
- **`embedder.py`** - Google Gemini embedding wrapper
- **`vector_store.py`** - Chroma vector database interface  
- **`retriever.py`** - Document retrieval logic
- **`gates.py`** - Relevance filtering with similarity scores and LLM judgment
- **`generator.py`** - Answer generation with Gemini models
- **`qa_fallback.py`** - Fallback system for handling edge cases
- **`pdf_loader.py`** - PDF text extraction utilities
- **`chunker.py`** - Document chunking strategies
- **`models.py`** - Data classes and type definitions

### Multi-Stage Retrieval Flow
1. **Primary retrieval**: Semantic search from PDF documents
2. **Score gate**: Filter by similarity threshold (configurable)
3. **LLM relevance gate**: Secondary relevance check with confidence scoring
4. **Partial answer mode**: Handle cases with limited but useful information
5. **QA fallback**: Use curated Q&A pairs when main retrieval fails

### Key Features
- **Auto-healing index**: Automatically rebuilds document index if unhealthy
- **Multi-mode generation**: Normal, partial, and fallback answer modes with natural language formatting
- **Thai language focus**: All responses in Thai with specialized formatting
- **Topic isolation**: Smart conversation history management that detects topic changes and limits cross-contamination
- **Intelligent memory**: Conversation history with semantic similarity-based topic detection
- **Comprehensive logging**: Detailed request/response logging with topic change tracking in `logs/app.log`

## Configuration
Primary configuration through `.env` file (copy from `.env.example`):
- **Models**: `GEN_MODEL=gemini-2.5-flash`, `EMBED_MODEL=gemini-embedding-001`  
- **Retrieval**: `TOP_K`, `SIM_THRESHOLD`, `GATE_CONF_THRESHOLD`
- **Fallback**: `QA_TOP_K`, `QA_SIM_THRESHOLD`
- **Partial answers**: `PARTIAL_ENABLED`, `PARTIAL_MIN_SCORE`, `PARTIAL_MIN_CHUNKS`
- **LLM behavior**: `LLM_TEMPERATURE=0.0`, `LLM_TOP_P=1.0`
- **Topic isolation**: `TOPIC_CHANGE_THRESHOLD=0.3`, `MAX_HISTORY_FOR_NEW_TOPIC=2`, `TOPIC_ISOLATION_ENABLED=true`

## File Structure
- `app.py` - Streamlit UI with custom Thai styling
- `data/pdfs/` - Source PDF documents for indexing
- `data/json/qa_prompt.json` - Curated Q&A pairs for fallback
- `storage/chroma/` - Vector database persistence
- `scripts/` - Utility scripts for indexing and evaluation
- `tests/` - Comprehensive test suite
- `reports/` - Benchmark and evaluation outputs

## Development Notes
- The system requires a valid `GEMINI_API_KEY` to function
- PDF documents are automatically chunked and indexed with metadata
- The relevance gate uses both similarity scores and LLM confidence ratings, with topic history awareness
- Topic change detection uses semantic embeddings to identify conversation topic shifts
- When topic changes are detected, conversation history is automatically truncated to prevent context contamination
- All user interactions are logged with detailed retrieval metrics and topic detection information
- The UI includes automatic index health monitoring and manual rebuild options

## Multi-Topic Chat Improvements (v2.1)
- **Smart History Management**: Automatically detects topic changes using semantic similarity
- **Context Isolation**: Limits conversation history when switching topics to prevent answer format confusion
- **Enhanced Logging**: Track topic changes and history truncation decisions
- **Natural Formatting**: Removed forced structured responses in partial/fallback modes
- **Configurable Thresholds**: Tune topic detection sensitivity and history limits via environment variables