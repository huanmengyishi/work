from __future__ import annotations

import shutil
import json
import sys
import tempfile
from pathlib import Path

from .base import ToolResult, run_command, truncate_text
from .pathsafe import resolve_project_path


TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".gd",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".sh",
    ".sql",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
WORD_SUFFIXES = {".docx", ".doc", ".odt", ".rtf"}
AI_TOOLS_PARSER = Path.home() / ".local" / "share" / "ai-tools" / "app" / "parser.py"
AI_TOOLS_LAUNCHER = Path.home() / ".local" / "bin" / "ai-parser"


class DocumentTool:
    def __init__(self, cwd: Path, timeout: int = 180) -> None:
        self.cwd = cwd
        self.timeout = timeout

    def parse(self, path: str, ocr: bool = True) -> ToolResult:
        file_path = self._resolve(path)
        if not file_path.exists():
            return ToolResult(False, "", f"file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix in {".pdf", *IMAGE_SUFFIXES, *WORD_SUFFIXES}:
            ai_tools_result = self._parse_with_ai_tools(file_path)
            if ai_tools_result.ok:
                return ai_tools_result
        if suffix in TEXT_SUFFIXES:
            return self._parse_text(file_path)
        if suffix == ".pdf":
            return self._parse_pdf(file_path, ocr=ocr)
        if suffix in IMAGE_SUFFIXES:
            return self._parse_image(file_path)
        if suffix in WORD_SUFFIXES:
            return ToolResult(False, "", "word parsing requires ai-parser dependencies or pandoc")
        return ToolResult(False, "", f"unsupported document type: {suffix or '<none>'}")

    def _resolve(self, path: str) -> Path:
        return resolve_project_path(self.cwd, path, require_file=True)

    def _parse_text(self, file_path: Path) -> ToolResult:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(
            True,
            markdown_document(file_path, truncate_text(text)),
            data={"path": str(file_path), "parser": "text"},
        )

    def _parse_pdf(self, file_path: Path, ocr: bool) -> ToolResult:
        if shutil.which("pdftotext"):
            result = run_command(
                ["pdftotext", "-layout", str(file_path), "-"],
                cwd=self.cwd,
                timeout=self.timeout,
            )
            if result.ok and result.output.strip():
                return ToolResult(
                    True,
                    markdown_document(file_path, truncate_text(result.output)),
                    data={"path": str(file_path), "parser": "pdftotext"},
                )
        if ocr:
            return self._ocr_pdf(file_path)
        return ToolResult(False, "", "PDF parsing failed and OCR is disabled")

    def _parse_with_ai_tools(self, file_path: Path) -> ToolResult:
        if not AI_TOOLS_LAUNCHER.exists() and not AI_TOOLS_PARSER.exists():
            return ToolResult(False, "", f"ai-parser not found: {AI_TOOLS_PARSER}")
        with tempfile.TemporaryDirectory(prefix="deep-agent-ai-parser-") as tmp:
            output_dir = Path(tmp)
            command = [str(AI_TOOLS_LAUNCHER)] if AI_TOOLS_LAUNCHER.exists() else [sys.executable, str(AI_TOOLS_PARSER)]
            result = run_command(
                [*command, str(file_path), "-o", str(output_dir)],
                cwd=self.cwd,
                timeout=self.timeout,
            )
            if not result.ok:
                return result
            json_files = sorted(output_dir.glob("*.json"))
            text_files = sorted(output_dir.glob("*.txt"))
            metadata = {"path": str(file_path), "parser": "ai-parser"}
            text = ""
            if json_files:
                try:
                    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
                    text = str(payload.get("text") or "")
                    if isinstance(payload.get("metadata"), dict):
                        metadata.update(payload["metadata"])
                    metadata["kind"] = payload.get("kind")
                except json.JSONDecodeError:
                    text = ""
            if not text and text_files:
                text = text_files[0].read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return ToolResult(False, result.output, "ai-parser produced no text", data=metadata)
            return ToolResult(
                True,
                markdown_document(file_path, truncate_text(text)),
                data=metadata,
            )

    def _parse_image(self, file_path: Path) -> ToolResult:
        if not shutil.which("tesseract"):
            return ToolResult(False, "", "tesseract is not installed")
        result = run_command(
            ["tesseract", str(file_path), "stdout", "-l", "eng+chi_sim"],
            cwd=self.cwd,
            timeout=self.timeout,
        )
        if not result.ok:
            result = run_command(
                ["tesseract", str(file_path), "stdout"],
                cwd=self.cwd,
                timeout=self.timeout,
            )
        if not result.ok:
            return result
        return ToolResult(
            True,
            markdown_document(file_path, truncate_text(result.output)),
            data={"path": str(file_path), "parser": "tesseract"},
        )

    def _ocr_pdf(self, file_path: Path) -> ToolResult:
        if not shutil.which("magick") and not shutil.which("convert"):
            return ToolResult(False, "", "PDF OCR requires ImageMagick magick/convert")
        if not shutil.which("tesseract"):
            return ToolResult(False, "", "PDF OCR requires tesseract")
        converter = shutil.which("magick") or shutil.which("convert")
        assert converter is not None
        with tempfile.TemporaryDirectory(prefix="deep-agent-ocr-") as tmp:
            tmpdir = Path(tmp)
            out_pattern = tmpdir / "page.png"
            cmd = [converter]
            if Path(converter).name == "magick":
                cmd.extend([str(file_path), str(out_pattern)])
            else:
                cmd.extend(["-density", "180", str(file_path), str(out_pattern)])
            convert_result = run_command(cmd, cwd=self.cwd, timeout=self.timeout)
            if not convert_result.ok:
                return convert_result
            parts = []
            for image in sorted(tmpdir.glob("page*.png")):
                ocr_result = self._parse_image(image)
                if ocr_result.ok and ocr_result.output.strip():
                    parts.append(f"## {image.name}\n\n{ocr_result.output}")
            if not parts:
                return ToolResult(False, "", "OCR produced no text")
            return ToolResult(
                True,
                markdown_document(file_path, truncate_text("\n\n".join(parts))),
                data={"path": str(file_path), "parser": "pdf-ocr"},
            )


def markdown_document(path: Path, content: str) -> str:
    return f"# Parsed Document\n\n- Source: `{path}`\n\n---\n\n{content.strip()}\n"
