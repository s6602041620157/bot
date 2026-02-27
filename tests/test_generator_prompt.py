from rag.generator import build_generation_prompt
from rag.models import RetrievedChunk


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="c1",
            file_name="a.pdf",
            page=1,
            text="ข้อมูลตัวอย่าง",
            score=0.9,
        )
    ]


def test_build_prompt_partial_mode_contains_required_sections() -> None:
    prompt = build_generation_prompt(
        query="ขั้นตอนการถอนวิชาเรียน",
        chunks=_sample_chunks(),
        history=[],
        mode="partial",
        style_policy="auto",
    )
    assert "## ข้อมูลที่พบ" in prompt
    assert "## คำถามแนะนำต่อ" in prompt
    assert "## สิ่งที่ยังไม่ชัดเจน" not in prompt


def test_build_prompt_auto_style_contains_format_rules() -> None:
    prompt = build_generation_prompt(
        query="เปรียบเทียบหลักสูตร A และ B",
        chunks=_sample_chunks(),
        history=[],
        mode="normal",
        style_policy="auto",
    )
    assert "ใช้ลำดับเลข 1,2,3" in prompt
    assert "ใช้ตาราง Markdown" in prompt


def test_build_prompt_list_style_is_forced() -> None:
    prompt = build_generation_prompt(
        query="การลงทะเบียนเรียน",
        chunks=_sample_chunks(),
        history=[],
        mode="normal",
        style_policy="list",
    )
    assert "รายการลำดับเลขเสมอ" in prompt
