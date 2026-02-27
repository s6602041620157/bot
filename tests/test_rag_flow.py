from pathlib import Path
from threading import Thread
import time

from rag.config import REFUSAL_MESSAGE, Settings
from rag.models import DocumentChunk
from rag.models import GateDecision, RAGAnswer, RetrievedChunk
from rag.pipeline import RAGPipeline


class FakeEmbedder:
    pass


class FakeVectorStore:
    def __init__(self) -> None:
        self._count = 2

    def count(self) -> int:
        return self._count

    def get_all_metadata(self):
        return [{"file_name": "a.pdf"}, {"file_name": "b.pdf"}]


class RaceVectorStore(FakeVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.busy_reset = False

    def reset(self) -> None:
        self.busy_reset = True
        time.sleep(0.08)
        self.busy_reset = False

    def upsert_chunks(self, chunks, vectors) -> None:  # noqa: ANN001
        return None


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class FakeQAFallback:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class FakeGate:
    def __init__(self, decision: GateDecision) -> None:
        self.decision = decision

    def judge_relevance(self, query: str, chunks: list[RetrievedChunk]) -> GateDecision:
        return self.decision


class FakeGenerator:
    def __init__(self) -> None:
        self.last_mode: str | None = None

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
        mode: str = "normal",
        style_policy: str = "auto",
    ) -> RAGAnswer:
        self.last_mode = mode
        return RAGAnswer(
            answer_text=f"คำตอบทดสอบ ({mode})",
            citations=chunks,
            grounded=True,
            refusal=False,
        )


class SequencedFakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs[:]
        self.calls = 0
        self.modes: list[str] = []

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
        mode: str = "normal",
        style_policy: str = "auto",
    ) -> RAGAnswer:
        self.calls += 1
        self.modes.append(mode)
        text = self.outputs.pop(0) if self.outputs else "คำตอบทดสอบ"
        return RAGAnswer(
            answer_text=text,
            citations=chunks,
            grounded=text != REFUSAL_MESSAGE,
            refusal=text == REFUSAL_MESSAGE,
        )


def _settings(tmp_path: Path) -> Settings:
    log_file = tmp_path / "logs" / "app.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        gemini_api_key="test",
        pdf_dir=tmp_path / "data/pdfs",
        chroma_dir=tmp_path / "storage/chroma",
        log_file=log_file,
    )


def test_ask_uses_fallback_when_pdf_score_gate_fails(tmp_path: Path) -> None:
    pdf_chunks = [RetrievedChunk("c1", "a.pdf", 1, "x", 0.2)]
    qa_chunks = [RetrievedChunk("qa:1", "คู่มือนักศึกษา", 0, "คำตอบจาก qa", 0.91)]
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(pdf_chunks),
        qa_fallback=FakeQAFallback(qa_chunks),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=generator,
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is False
    assert generator.last_mode == "fallback"
    assert result.citations[0].chunk_id == "qa:1"


def test_ask_uses_partial_when_llm_gate_rejects_but_evidence_is_strong(tmp_path: Path) -> None:
    pdf_chunks = [
        RetrievedChunk("c1", "a.pdf", 1, "ข้อมูลสำคัญ 1", 0.81),
        RetrievedChunk("c2", "a.pdf", 2, "ข้อมูลสำคัญ 2", 0.76),
    ]
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(pdf_chunks),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(False, 0.2, "insufficient", [])),
        generator=generator,
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is False
    assert result.grounded is True
    assert generator.last_mode == "partial"


def test_ask_refuses_when_pdf_and_fallback_are_insufficient(tmp_path: Path) -> None:
    pdf_chunks = [RetrievedChunk("c1", "a.pdf", 1, "x", 0.2)]
    qa_chunks = [RetrievedChunk("qa:1", "คู่มือนักศึกษา", 0, "คำตอบไม่เกี่ยวข้อง", 0.4)]
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(pdf_chunks),
        qa_fallback=FakeQAFallback(qa_chunks),
        relevance_gate=FakeGate(GateDecision(False, 0.1, "insufficient", [])),
        generator=generator,
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is True
    assert result.answer_text == REFUSAL_MESSAGE


def test_ask_returns_normal_answer_when_llm_gate_passes(tmp_path: Path) -> None:
    pdf_chunks = [
        RetrievedChunk("c1", "a.pdf", 1, "ข้อมูลสำคัญ", 0.91),
        RetrievedChunk("c2", "a.pdf", 2, "ข้อมูลเสริม", 0.88),
    ]
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(pdf_chunks),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=generator,
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is False
    assert result.grounded is True
    assert result.citations[0].chunk_id == "c1"
    assert generator.last_mode == "normal"


def test_ask_retries_normal_mode_after_refusal_then_recovers(tmp_path: Path) -> None:
    pdf_chunks = [
        RetrievedChunk("c1", "a.pdf", 1, "ข้อมูลสำคัญ", 0.91),
        RetrievedChunk("c2", "a.pdf", 2, "ข้อมูลเสริม", 0.88),
    ]
    generator = SequencedFakeGenerator([REFUSAL_MESSAGE, "คำตอบหลัง retry"])

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(pdf_chunks),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=generator,
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is False
    assert result.answer_text == "คำตอบหลัง retry"
    assert generator.calls == 2
    assert generator.modes == ["normal", "normal"]


def test_ask_downgrades_to_partial_when_normal_stays_refusal(tmp_path: Path) -> None:
    pdf_chunks = [
        RetrievedChunk("c1", "a.pdf", 1, "ข้อมูลสำคัญ", 0.91),
        RetrievedChunk("c2", "a.pdf", 2, "ข้อมูลเสริม", 0.88),
    ]
    generator = SequencedFakeGenerator([REFUSAL_MESSAGE, REFUSAL_MESSAGE, "คำตอบแบบ partial"])

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(pdf_chunks),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=generator,
    )

    result = pipeline.ask("ถาม", history=[])
    assert result.refusal is False
    assert result.answer_text == "คำตอบแบบ partial"
    assert generator.calls == 3
    assert generator.modes == ["normal", "normal", "partial"]


def test_ensure_pdf_index_healthy_triggers_reindex_when_index_is_empty(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class EmptyVectorStore(FakeVectorStore):
        def count(self) -> int:
            return 0

        def get_all_metadata(self):
            return []

    pipeline = RAGPipeline(
        settings=settings,
        embedder=FakeEmbedder(),
        vector_store=EmptyVectorStore(),
        retriever=FakeRetriever([]),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(False, 0.0, "none", [])),
        generator=FakeGenerator(),
    )

    called = {"count": 0}

    def _fake_index(pdf_dir: str | None = None, reset: bool = False):  # noqa: ANN001
        called["count"] += 1
        return {
            "indexed_files": 5,
            "indexed_chunks": 100,
            "collection_count": 100,
            "document_count": 5,
            "pdf_dir": str(pdf_dir or settings.pdf_dir),
        }

    pipeline.index_pdfs = _fake_index  # type: ignore[method-assign]
    status = pipeline.ensure_pdf_index_healthy(force=True)
    assert called["count"] == 1
    assert status["action"] == "reindexed"


def test_ensure_pdf_index_healthy_skips_when_index_is_healthy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pipeline = RAGPipeline(
        settings=settings,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever([]),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(False, 0.0, "none", [])),
        generator=FakeGenerator(),
    )

    called = {"count": 0}

    def _fake_index(pdf_dir: str | None = None, reset: bool = False):  # noqa: ANN001
        called["count"] += 1
        return {}

    pipeline.index_pdfs = _fake_index  # type: ignore[method-assign]
    status = pipeline.ensure_pdf_index_healthy(force=True)
    assert called["count"] == 0
    assert status["action"] == "none"
    assert status["healthy"] is True


def test_ask_is_serialized_against_index_reset(tmp_path: Path, monkeypatch) -> None:
    vector_store = RaceVectorStore()

    class RaceEmbedder(FakeEmbedder):
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] for _ in texts]

    class RaceRetriever(FakeRetriever):
        def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
            if vector_store.busy_reset:
                raise RuntimeError("race_detected")
            return [RetrievedChunk("c1", "a.pdf", 1, "ข้อมูลสำคัญ", 0.92)]

    pipeline = RAGPipeline(
        settings=_settings(tmp_path),
        embedder=RaceEmbedder(),
        vector_store=vector_store,
        retriever=RaceRetriever([]),
        qa_fallback=FakeQAFallback([]),
        relevance_gate=FakeGate(GateDecision(True, 0.9, "ok", ["c1"])),
        generator=FakeGenerator(),
    )

    pdf_path = tmp_path / "data/pdfs" / "dummy.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr("rag.pipeline.list_pdf_files", lambda _directory: [pdf_path])
    monkeypatch.setattr("rag.pipeline.extract_pdf_pages", lambda _pdf: [(1, "ข้อมูลสำหรับเทสต์")])
    monkeypatch.setattr(
        "rag.pipeline.build_document_chunks",
        lambda pdf_path, page_texts: [
            DocumentChunk(
                id=f"{pdf_path.name}:p1:c1",
                file_name=pdf_path.name,
                page=1,
                text=page_texts[0][1],
                token_count=1,
                source_path=str(pdf_path),
                metadata={},
            )
        ],
    )

    index_error: list[Exception] = []

    def _run_index() -> None:
        try:
            pipeline.index_pdfs(pdf_dir=str(pdf_path.parent), reset=True)
        except Exception as exc:  # pragma: no cover
            index_error.append(exc)

    thread = Thread(target=_run_index)
    thread.start()
    time.sleep(0.01)
    answer = pipeline.ask("ถาม", history=[])
    thread.join()

    assert not index_error
    assert answer.refusal is False
