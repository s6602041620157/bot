from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

def _get_secret(key: str, default: str = "") -> str:
    """Read from env var first, then fall back to st.secrets."""
    val = os.getenv(key, "").strip()
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return default

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
    top_k: int = 8
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
    gen_retry_on_refusal: int = 0
    auto_heal_index: bool = True
    auto_heal_min_docs: int = 1
    skip_llm_gate: bool = False
    gate_skip_score: float = 0.75

    def ensure_paths(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)



def load_settings() -> Settings:
    load_dotenv(override=False)

    api_key = _get_secret("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Set it in .env (local) or Streamlit secrets (cloud)."
        )
    settings = Settings(
        gemini_api_key=api_key,
        gen_model=_get_secret("GEN_MODEL", "gemini-2.5-flash"),
        embed_model=_get_secret("EMBED_MODEL", "gemini-embedding-001"),
        pdf_dir=Path(_get_secret("PDF_DIR", "data/pdfs")),
        chroma_dir=Path(_get_secret("CHROMA_DIR", "storage/chroma")),
        top_k=int(_get_secret("TOP_K", "8")),
        sim_threshold=float(_get_secret("SIM_THRESHOLD", "0.55")),
        gate_conf_threshold=float(_get_secret("GATE_CONF_THRESHOLD", "0.65")),
        collection_name=_get_secret("CHROMA_COLLECTION", "pdf_rag_chunks"),
        qa_json_path=Path(_get_secret("QA_JSON_PATH", "data/json/qa_prompt.json")),
        qa_collection_name=_get_secret("QA_COLLECTION", "qa_prompt_chunks"),
        qa_top_k=int(_get_secret("QA_TOP_K", "5")),
        qa_sim_threshold=float(_get_secret("QA_SIM_THRESHOLD", "0.60")),
        partial_enabled=_env_bool(_get_secret("PARTIAL_ENABLED") or None, default=True),
        partial_min_score=float(_get_secret("PARTIAL_MIN_SCORE", "0.68")),
        partial_min_chunks=int(_get_secret("PARTIAL_MIN_CHUNKS", "2")),
        answer_style_policy=_get_secret("ANSWER_STYLE_POLICY", "auto") or "auto",
        log_file=Path(_get_secret("LOG_FILE", "logs/app.log")),
        memory_turns=int(_get_secret("MEMORY_TURNS", "6")),
        llm_temperature=float(_get_secret("LLM_TEMPERATURE", "0.0")),
        llm_top_p=float(_get_secret("LLM_TOP_P", "1.0")),
        gen_retry_on_refusal=max(0, int(_get_secret("GEN_RETRY_ON_REFUSAL", "0"))),
        auto_heal_index=_env_bool(_get_secret("AUTO_HEAL_INDEX") or None, default=True),
        auto_heal_min_docs=max(1, int(_get_secret("AUTO_HEAL_MIN_DOCS", "1"))),
        skip_llm_gate=_env_bool(_get_secret("SKIP_LLM_GATE") or None, default=False),
        gate_skip_score=float(_get_secret("GATE_SKIP_SCORE", "0.75")),
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
