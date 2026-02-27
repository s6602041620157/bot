from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REFUSAL_MESSAGE = (
    "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่ให้มา "
    "กรุณาระบุคำถามให้เฉพาะเจาะจงขึ้น"
)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gen_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    pdf_dir: Path = Path("data/pdfs")
    chroma_dir: Path = Path("storage/chroma")
    top_k: int = 12
    sim_threshold: float = 0.55
    gate_conf_threshold: float = 0.65
    collection_name: str = "pdf_rag_chunks"
    qa_json_path: Path = Path("data/json/qa_prompt.json")
    qa_collection_name: str = "qa_prompt_chunks"
    qa_top_k: int = 5
    qa_sim_threshold: float = 0.60
    partial_enabled: bool = True
    partial_min_score: float = 0.68
    partial_min_chunks: int = 2
    answer_style_policy: str = "auto"
    log_file: Path = Path("logs/app.log")
    memory_turns: int = 6
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    gen_retry_on_refusal: int = 1
    auto_heal_index: bool = True
    auto_heal_min_docs: int = 1

    def ensure_paths(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)



def load_settings() -> Settings:
    load_dotenv(override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    settings = Settings(
        gemini_api_key=api_key,
        gen_model=os.getenv("GEN_MODEL", "gemini-2.5-flash").strip(),
        embed_model=os.getenv("EMBED_MODEL", "gemini-embedding-001").strip(),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdfs")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "storage/chroma")),
        top_k=int(os.getenv("TOP_K", "12")),
        sim_threshold=float(os.getenv("SIM_THRESHOLD", "0.55")),
        gate_conf_threshold=float(os.getenv("GATE_CONF_THRESHOLD", "0.65")),
        collection_name=os.getenv("CHROMA_COLLECTION", "pdf_rag_chunks").strip(),
        qa_json_path=Path(os.getenv("QA_JSON_PATH", "data/json/qa_prompt.json")),
        qa_collection_name=os.getenv("QA_COLLECTION", "qa_prompt_chunks").strip(),
        qa_top_k=int(os.getenv("QA_TOP_K", "5")),
        qa_sim_threshold=float(os.getenv("QA_SIM_THRESHOLD", "0.60")),
        partial_enabled=_env_bool(os.getenv("PARTIAL_ENABLED"), default=True),
        partial_min_score=float(os.getenv("PARTIAL_MIN_SCORE", "0.68")),
        partial_min_chunks=int(os.getenv("PARTIAL_MIN_CHUNKS", "2")),
        answer_style_policy=os.getenv("ANSWER_STYLE_POLICY", "auto").strip() or "auto",
        log_file=Path(os.getenv("LOG_FILE", "logs/app.log")),
        memory_turns=int(os.getenv("MEMORY_TURNS", "6")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        gen_retry_on_refusal=max(0, int(os.getenv("GEN_RETRY_ON_REFUSAL", "1"))),
        auto_heal_index=_env_bool(os.getenv("AUTO_HEAL_INDEX"), default=True),
        auto_heal_min_docs=max(1, int(os.getenv("AUTO_HEAL_MIN_DOCS", "1"))),
    )
    settings.ensure_paths()
    return settings


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default
