from __future__ import annotations

import json
import re
from typing import Any

from google import genai

from .models import GateDecision, RetrievedChunk



def apply_score_gate(chunks: list[RetrievedChunk], threshold: float) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.score >= threshold]



def parse_gate_response_json(text: str) -> GateDecision:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return GateDecision(is_relevant=False, confidence=0.0, reason="invalid_json", kept_chunk_ids=[])

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return GateDecision(is_relevant=False, confidence=0.0, reason="json_decode_error", kept_chunk_ids=[])

    raw_relevant = parsed.get("is_relevant", False)
    if isinstance(raw_relevant, bool):
        is_relevant = raw_relevant
    elif isinstance(raw_relevant, str):
        is_relevant = raw_relevant.strip().lower() in {"true", "1", "yes"}
    else:
        is_relevant = bool(raw_relevant)

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(parsed.get("reason", ""))

    raw_ids = parsed.get("kept_chunk_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []
    kept_ids = [str(x) for x in raw_ids if str(x).strip()]

    return GateDecision(
        is_relevant=is_relevant,
        confidence=confidence,
        reason=reason,
        kept_chunk_ids=kept_ids,
    )


class LLMRelevanceGate:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0, top_p: float = 1.0) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def judge_relevance(self, query: str, chunks: list[RetrievedChunk]) -> GateDecision:
        if not chunks:
            return GateDecision(
                is_relevant=False,
                confidence=0.0,
                reason="no_chunks",
                kept_chunk_ids=[],
            )

        context_lines = []
        for chunk in chunks:
            snippet = chunk.text[:500]
            context_lines.append(
                f"- id={chunk.chunk_id} | file={chunk.file_name} | page={chunk.page} | score={chunk.score:.3f} | text={snippet}"
            )

        prompt = (
            "You are a strict relevance judge for Retrieval-Augmented Generation. "
            "Assess whether the provided evidence is sufficient to answer the user query. "
            "Return JSON only with schema: "
            "{\"is_relevant\": bool, \"confidence\": float (0-1), \"reason\": str, \"kept_chunk_ids\": list[str]}. "
            "If evidence is not sufficient, set is_relevant=false and keep list empty or minimal.\n\n"
            f"Query:\n{query}\n\n"
            "Candidate chunks:\n"
            + "\n".join(context_lines)
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature, "top_p": self.top_p},
        )
        text = _response_text(response)
        decision = parse_gate_response_json(text)

        valid_ids = {c.chunk_id for c in chunks}
        decision.kept_chunk_ids = [cid for cid in decision.kept_chunk_ids if cid in valid_ids]
        return decision


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    if hasattr(response, "candidates"):
        parts: list[str] = []
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                candidate_text = getattr(part, "text", None)
                if candidate_text:
                    parts.append(candidate_text)
        if parts:
            return "\n".join(parts)

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        return json.dumps(dumped, ensure_ascii=False)

    return ""
