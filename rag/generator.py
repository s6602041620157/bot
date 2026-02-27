from __future__ import annotations

import json
from typing import Any

from google import genai

from .config import REFUSAL_MESSAGE
from .models import RAGAnswer, RetrievedChunk


class GeminiAnswerGenerator:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0, top_p: float = 1.0) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict],
        mode: str = "normal",
        style_policy: str = "auto",
    ) -> RAGAnswer:
        if not chunks:
            return RAGAnswer(answer_text=REFUSAL_MESSAGE, citations=[], grounded=False, refusal=True)

        prompt = build_generation_prompt(
            query=query,
            chunks=chunks,
            history=history,
            mode=mode,
            style_policy=style_policy,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature, "top_p": self.top_p},
        )
        answer_text = _response_text(response).strip() or REFUSAL_MESSAGE

        refusal = answer_text == REFUSAL_MESSAGE
        return RAGAnswer(
            answer_text=answer_text,
            citations=chunks,
            grounded=not refusal,
            refusal=refusal,
        )


def build_generation_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict],
    mode: str = "normal",
    style_policy: str = "auto",
) -> str:
    history_text = _format_history(history)
    context_text = "\n".join(
        f"[chunk_id={c.chunk_id}] file={c.file_name} page={c.page}\n{c.text}"
        for c in chunks
    )
    mode_instruction = _mode_instruction(mode)
    style_instruction = _style_instruction(style_policy)

    return (
        "คุณคือผู้ช่วยตอบคำถามจากเอกสารเท่านั้น\n"
        "ข้อกำหนดหลัก:\n"
        "1) ตอบเป็นภาษาไทยเท่านั้น\n"
        "2) ใช้เฉพาะข้อมูลจาก CONTEXT ที่ให้มา\n"
        "3) ห้ามเดาหรือเพิ่มข้อมูลจากภายนอกโดยเด็ดขาด\n"
        "4) ห้ามแสดงแหล่งที่มาในเนื้อคำตอบ\n"
        "5) ถ้าข้อมูลไม่พอจริง ๆ ให้ตอบข้อความนี้เท่านั้น: "
        f"\"{REFUSAL_MESSAGE}\"\n"
        "6) ถ้าคำถามกว้าง ให้ตอบเท่าที่พบก่อน แล้วชวนผู้ใช้ถามต่อให้เจาะจงขึ้น\n\n"
        f"MODE:\n{mode_instruction}\n\n"
        f"FORMAT_POLICY:\n{style_instruction}\n\n"
        f"CHAT_HISTORY:\n{history_text}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{query}\n"
    )


def _mode_instruction(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"partial", "fallback"}:
        return (
            "โหมดนี้เป็นการตอบแบบมีเงื่อนไข: ให้ตอบเฉพาะข้อมูลที่ยืนยันได้จาก CONTEXT\n"
            "บังคับโครงสร้างคำตอบเป็น Markdown ตามหัวข้อต่อไปนี้เท่านั้น:\n"
            "## ข้อมูลที่พบ\n"
            "## คำตอบ\n"
            "## คำถามแนะนำต่อ\n"
            "หัวข้อสุดท้ายต้องมีคำถามต่อยอด 2-3 ข้อในรูปแบบ bullet"
        )
    return (
        "โหมดปกติ: ตอบให้ครบถ้วนตามหลักฐานใน CONTEXT\n"
        "ถ้าคำถามกว้างหรือกำกวม ให้สรุปภาพรวมก่อนและปิดท้ายด้วยคำถามแนะนำต่อ 1-2 ข้อ"
    )


def _style_instruction(style_policy: str) -> str:
    policy = style_policy.strip().lower()
    if policy == "list":
        return "จัดรูปแบบคำตอบเป็นรายการลำดับเลขเสมอ"
    if policy == "paragraph":
        return "จัดรูปแบบคำตอบเป็นย่อหน้าเสมอ"
    return (
        "เลือก format อัตโนมัติตามลักษณะคำถาม:\n"
        "- คำถามเชิงขั้นตอน/วิธีทำ: ใช้ลำดับเลข 1,2,3\n"
        "- คำถามเชิงอธิบาย/นิยาม: ใช้ย่อหน้าสั้นและ bullet สรุป\n"
        "- คำถามเชิงเปรียบเทียบ: ใช้ตาราง Markdown"
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(empty)"
    lines: list[str] = []
    for item in history:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(empty)"


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
