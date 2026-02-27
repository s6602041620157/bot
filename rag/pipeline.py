from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import RLock
from typing import Any

from .chunker import build_document_chunks
from .config import REFUSAL_MESSAGE, Settings, load_settings
from .embedder import GeminiEmbedder
from .gates import LLMRelevanceGate, apply_score_gate
from .generator import GeminiAnswerGenerator
from .models import GateDecision, RAGAnswer, RetrievedChunk
from .pdf_loader import extract_pdf_pages, list_pdf_files
from .qa_fallback import QAFallbackRetriever
from .retriever import Retriever
from .vector_store import ChromaVectorStore

_PIPELINE_LOCK = RLock()


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: GeminiEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        retriever: Retriever | None = None,
        qa_fallback: QAFallbackRetriever | None = None,
        relevance_gate: LLMRelevanceGate | None = None,
        generator: GeminiAnswerGenerator | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.logger = _build_logger(self.settings.log_file)
        self._lock = _PIPELINE_LOCK
        self._auto_heal_checked = False
        self._last_auto_heal_status: dict[str, Any] = {
            "action": "not_checked",
            "healthy": False,
            "index_stats": {"chunk_count": 0, "document_count": 0},
            "indexed_files": 0,
            "indexed_chunks": 0,
            "elapsed_ms": 0.0,
            "error": None,
        }

        self.embedder = embedder or GeminiEmbedder(
            api_key=self.settings.gemini_api_key,
            model=self.settings.embed_model,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.collection_name,
        )
        self.retriever = retriever or Retriever(self.embedder, self.vector_store)
        self.qa_fallback = qa_fallback or QAFallbackRetriever(
            embedder=self.embedder,
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.qa_collection_name,
            qa_json_path=self.settings.qa_json_path,
            logger=self.logger,
        )
        self.relevance_gate = relevance_gate or LLMRelevanceGate(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gen_model,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
        )
        self.generator = generator or GeminiAnswerGenerator(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gen_model,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
        )
        self._last_auto_heal_status["index_stats"] = self.get_index_stats()

    def index_pdfs(self, pdf_dir: str | None = None, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            directory = Path(pdf_dir) if pdf_dir else self.settings.pdf_dir
            pdf_files = list_pdf_files(directory)

            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {directory}")

            if reset:
                self.vector_store.reset()

            total_chunks = 0
            indexed_docs = 0

            for pdf_path in pdf_files:
                pages = extract_pdf_pages(pdf_path)
                chunks = build_document_chunks(pdf_path=pdf_path, page_texts=pages)
                if not chunks:
                    continue

                vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
                self.vector_store.upsert_chunks(chunks, vectors)
                total_chunks += len(chunks)
                indexed_docs += 1

            stats = self.get_index_stats()
            result = {
                "indexed_files": indexed_docs,
                "indexed_chunks": total_chunks,
                "collection_count": stats["chunk_count"],
                "document_count": stats["document_count"],
                "pdf_dir": str(directory),
            }
            self.logger.info("index_complete %s", result)
            return result

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query=query, top_k=top_k)

    def retrieve_qa_fallback(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.qa_fallback:
            return []
        return self.qa_fallback.retrieve(query=query, top_k=top_k)

    def judge_relevance(self, query: str, chunks: list[RetrievedChunk]) -> GateDecision:
        return self.relevance_gate.judge_relevance(query=query, chunks=chunks)

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
        mode: str = "normal",
    ) -> RAGAnswer:
        return self.generator.generate_answer(
            query=query,
            chunks=chunks,
            history=history,
            mode=mode,
            style_policy=self.settings.answer_style_policy,
        )

    def ask(self, query: str, history: list[dict]) -> RAGAnswer:
        with self._lock:
            start_time = time.perf_counter()
            initial_route = "refusal"
            final_route = "refusal"
            decision: GateDecision | None = None
            retry_count = 0

            retrieved = self.retrieve(query=query, top_k=self.settings.top_k)
            scored = apply_score_gate(retrieved, threshold=self.settings.sim_threshold)
            qa_retrieved: list[RetrievedChunk] = []
            qa_scored: list[RetrievedChunk] = []

            if scored:
                decision = self.judge_relevance(query=query, chunks=scored)
                if decision.is_relevant and decision.confidence >= self.settings.gate_conf_threshold:
                    selected = self._select_from_decision(scored, decision)
                    initial_route = "normal"
                    answer, retry_count = self._generate_with_retry(
                        query=query,
                        chunks=selected,
                        history=history,
                        mode="normal",
                    )
                    final_route = "normal"
                    if answer.refusal and self._can_partial_answer(scored):
                        partial_chunks = self._select_partial_chunks(scored)
                        partial_answer, partial_retry = self._generate_with_retry(
                            query=query,
                            chunks=partial_chunks,
                            history=history,
                            mode="partial",
                        )
                        retry_count += partial_retry
                        answer = partial_answer
                        final_route = "partial"
                    self._log_query(
                        query=query,
                        retrieved=retrieved,
                        qa_retrieved=qa_retrieved,
                        decision=decision,
                        answer=answer,
                        elapsed=time.perf_counter() - start_time,
                        initial_route=initial_route,
                        final_route=final_route,
                        retry_count=retry_count,
                    )
                    return answer

                if self._can_partial_answer(scored):
                    selected = self._select_partial_chunks(scored)
                    initial_route = "partial"
                    answer, retry_count = self._generate_with_retry(
                        query=query,
                        chunks=selected,
                        history=history,
                        mode="partial",
                    )
                    final_route = "partial"
                    self._log_query(
                        query=query,
                        retrieved=retrieved,
                        qa_retrieved=qa_retrieved,
                        decision=decision,
                        answer=answer,
                        elapsed=time.perf_counter() - start_time,
                        initial_route=initial_route,
                        final_route=final_route,
                        retry_count=retry_count,
                    )
                    return answer

            qa_retrieved = self.retrieve_qa_fallback(query=query, top_k=self.settings.qa_top_k)
            qa_scored = apply_score_gate(qa_retrieved, threshold=self.settings.qa_sim_threshold)
            if qa_scored:
                initial_route = "fallback"
                final_route = "fallback"
                answer, retry_count = self._generate_with_retry(
                    query=query,
                    chunks=qa_scored[:5],
                    history=history,
                    mode="fallback",
                )
                self._log_query(
                    query=query,
                    retrieved=retrieved,
                    qa_retrieved=qa_retrieved,
                    decision=decision,
                    answer=answer,
                    elapsed=time.perf_counter() - start_time,
                    initial_route=initial_route,
                    final_route=final_route,
                    retry_count=retry_count,
                )
                return answer

            citations = scored[:3] if scored else qa_scored[:3]
            answer = self._refusal_answer(citations=citations)
            self._log_query(
                query=query,
                retrieved=retrieved,
                qa_retrieved=qa_retrieved,
                decision=decision,
                answer=answer,
                elapsed=time.perf_counter() - start_time,
                initial_route=initial_route,
                final_route=final_route,
                retry_count=retry_count,
            )
            return answer

    def has_index(self) -> bool:
        with self._lock:
            return self.vector_store.count() > 0

    def get_index_stats(self) -> dict[str, int]:
        with self._lock:
            chunk_count = self.vector_store.count()
            metadatas = self.vector_store.get_all_metadata()

            files = {
                str(meta.get("file_name"))
                for meta in metadatas
                if isinstance(meta, dict) and meta.get("file_name")
            }
            return {
                "chunk_count": int(chunk_count),
                "document_count": len(files),
            }

    def ensure_pdf_index_healthy(self, force: bool = False) -> dict[str, Any]:
        if not self.settings.auto_heal_index:
            status = {
                "action": "disabled",
                "healthy": self.has_index(),
                "index_stats": self.get_index_stats(),
                "indexed_files": 0,
                "indexed_chunks": 0,
                "elapsed_ms": 0.0,
                "error": None,
            }
            self._last_auto_heal_status = status
            return status

        with self._lock:
            if self._auto_heal_checked and not force:
                return dict(self._last_auto_heal_status)

            started = time.perf_counter()
            stats_before = self.get_index_stats()
            healthy = (
                stats_before["chunk_count"] > 0
                and stats_before["document_count"] >= self.settings.auto_heal_min_docs
            )

            if healthy:
                status = {
                    "action": "none",
                    "healthy": True,
                    "index_stats": stats_before,
                    "indexed_files": 0,
                    "indexed_chunks": 0,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "error": None,
                }
                self._auto_heal_checked = True
                self._last_auto_heal_status = status
                self.logger.info("auto_heal_skip healthy=%s stats=%s", True, stats_before)
                return dict(status)

            self.logger.info("auto_heal_start stats_before=%s", stats_before)
            try:
                result = self.index_pdfs(pdf_dir=str(self.settings.pdf_dir), reset=True)
                stats_after = self.get_index_stats()
                status = {
                    "action": "reindexed",
                    "healthy": stats_after["chunk_count"] > 0
                    and stats_after["document_count"] >= self.settings.auto_heal_min_docs,
                    "index_stats": stats_after,
                    "indexed_files": int(result.get("indexed_files", 0)),
                    "indexed_chunks": int(result.get("indexed_chunks", 0)),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "error": None,
                }
                self.logger.info(
                    "auto_heal_success elapsed_ms=%.2f result=%s",
                    status["elapsed_ms"],
                    status,
                )
            except Exception as exc:
                status = {
                    "action": "failed",
                    "healthy": False,
                    "index_stats": self.get_index_stats(),
                    "indexed_files": 0,
                    "indexed_chunks": 0,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "error": str(exc),
                }
                self.logger.exception("auto_heal_failed elapsed_ms=%.2f", status["elapsed_ms"])
            self._auto_heal_checked = True
            self._last_auto_heal_status = status
            return dict(status)

    def get_auto_heal_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_auto_heal_status)

    def _refusal_answer(self, citations: list[RetrievedChunk] | None = None) -> RAGAnswer:
        return RAGAnswer(
            answer_text=REFUSAL_MESSAGE,
            citations=citations or [],
            grounded=False,
            refusal=True,
        )

    def _select_from_decision(self, chunks: list[RetrievedChunk], decision: GateDecision) -> list[RetrievedChunk]:
        if not decision.kept_chunk_ids:
            return chunks[:5]
        selected = [chunk for chunk in chunks if chunk.chunk_id in set(decision.kept_chunk_ids)]
        return (selected or chunks)[:5]

    def _can_partial_answer(self, chunks: list[RetrievedChunk]) -> bool:
        if not self.settings.partial_enabled:
            return False
        strong_chunks = [chunk for chunk in chunks if chunk.score >= self.settings.partial_min_score]
        return len(strong_chunks) >= self.settings.partial_min_chunks

    def _select_partial_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        strong_chunks = [chunk for chunk in chunks if chunk.score >= self.settings.partial_min_score]
        selected = strong_chunks if strong_chunks else chunks
        return selected[:5]

    def _generate_with_retry(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
        mode: str,
    ) -> tuple[RAGAnswer, int]:
        answer = self.generate_answer(query=query, chunks=chunks, history=history, mode=mode)
        retries = 0
        while answer.refusal and retries < self.settings.gen_retry_on_refusal:
            retries += 1
            answer = self.generate_answer(query=query, chunks=chunks, history=history, mode=mode)
        return answer, retries

    def _log_query(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        qa_retrieved: list[RetrievedChunk],
        decision: GateDecision | None,
        answer: RAGAnswer,
        elapsed: float,
        initial_route: str,
        final_route: str,
        retry_count: int,
    ) -> None:
        index_stats = self.get_index_stats()
        route_status = "refusal_recovered" if initial_route != final_route and not answer.refusal else (
            "refusal_persisted" if answer.refusal else "stable"
        )
        self.logger.info(
            (
                "initial_route=%s final_route=%s route_status=%s query=%s elapsed_ms=%.2f retry_count=%s "
                "index_stats=%s retrieved_ids=%s scores=%s qa_ids=%s qa_scores=%s gate=%s refusal=%s"
            ),
            initial_route,
            final_route,
            route_status,
            query,
            elapsed * 1000,
            retry_count,
            index_stats,
            [r.chunk_id for r in retrieved],
            [round(r.score, 4) for r in retrieved],
            [r.chunk_id for r in qa_retrieved],
            [round(r.score, 4) for r in qa_retrieved],
            decision,
            answer.refusal,
        )


def _build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("rag_pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
