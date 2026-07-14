from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
from zipfile import ZipFile

from docx import Document


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_release_docx.py"
SPEC = importlib.util.spec_from_file_location("build_release_docx", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_release_docx_uses_versioned_source_title_and_release_metadata(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    destination = tmp_path / "guide.docx"
    source.write_text(
        "# DeepSeek Agent V3 使用说明（0.11.0）\n\n"
        "日期：2026-07-15\n\n1. 第一项\n2. 第二项\n\n## 新列表\n\n1. 重新从一开始\n",
        encoding="utf-8",
    )

    builder.markdown_to_docx(source, destination)

    reopened = Document(destination)
    properties = reopened.core_properties
    expected = datetime(2026, 7, 15, tzinfo=timezone.utc)
    assert properties.title == "DeepSeek Agent V3 使用说明（0.11.0）"
    assert properties.author == "Deep Agent"
    assert properties.last_modified_by == "Deep Agent"
    assert properties.created == expected
    assert properties.modified == expected
    assert properties.revision == 1
    paragraph_text = [paragraph.text for paragraph in reopened.paragraphs]
    assert "1. 第一项" in paragraph_text
    assert "2. 第二项" in paragraph_text
    assert "1. 重新从一开始" in paragraph_text
    with ZipFile(destination) as archive:
        core_xml = archive.read("docProps/core.xml").decode("utf-8")
    assert "2026-07-15T00:00:00Z" in core_xml
    assert "2013-" not in core_xml
