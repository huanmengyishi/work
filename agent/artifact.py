from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

import yaml

if TYPE_CHECKING:
    from .tools.base import ToolResult


DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES_HARD_LIMIT = 64 * 1024 * 1024
DEFAULT_MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES_HARD_LIMIT = 256 * 1024 * 1024
DEFAULT_MAX_DOCX_COMPRESSION_RATIO = 100.0
MAX_DOCX_COMPRESSION_RATIO_HARD_LIMIT = 1_000.0
MAX_DOCX_MEMBERS = 2_048
MAX_METADATA_KEYS = 128
ARTIFACT_VERIFICATION_SCHEMA_VERSION = 1
ARTIFACT_VERIFICATION_METADATA_KEY = "artifact_verification"
MANAGED_DOCUMENT_ARTIFACT_ID = "document-parse"

ARTIFACT_FORMATS = frozenset({"auto", "binary", "text", "json", "yaml", "python", "docx"})
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst", ".csv"})
_DOCX_REQUIRED_MEMBERS = frozenset({"[Content_Types].xml", "_rels/.rels", "word/document.xml"})
_WORD_TEXT_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCX_RECEIPT_CHECKS = frozenset(
    {
        "tool_result",
        "exists",
        "nonempty",
        "max_size",
        "content",
        "docx_zip_container",
        "docx_package_structure",
        "docx_zip_integrity",
        "docx_package_xml",
        "docx_nonempty_text",
    }
)


@dataclass(frozen=True)
class ArtifactSpec:
    """Bounded expectations for one managed artifact.

    ``path`` is an identity and format hint only.  Verification never opens it;
    callers must supply content or trustworthy metadata from a ToolResult.
    """

    artifact_id: str
    path: str
    format: str = "auto"
    min_bytes: int = 1
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    max_docx_uncompressed_bytes: int = DEFAULT_MAX_DOCX_UNCOMPRESSED_BYTES
    max_docx_compression_ratio: float = DEFAULT_MAX_DOCX_COMPRESSION_RATIO

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(self.artifact_id):
            raise ValueError("artifact_id must contain 1-128 ASCII letters, digits, '.', '_', or '-'")
        normalized_path = _bounded_relative_path(self.path)
        normalized_format = str(self.format).strip().lower()
        if normalized_format not in ARTIFACT_FORMATS:
            raise ValueError(f"unsupported artifact format: {self.format}")
        if not _bounded_integer(self.min_bytes, minimum=0, maximum=MAX_ARTIFACT_BYTES_HARD_LIMIT):
            raise ValueError("min_bytes is outside the supported range")
        if not _bounded_integer(self.max_bytes, minimum=1, maximum=MAX_ARTIFACT_BYTES_HARD_LIMIT):
            raise ValueError("max_bytes is outside the supported range")
        if self.min_bytes > self.max_bytes:
            raise ValueError("min_bytes cannot exceed max_bytes")
        if not _bounded_integer(
            self.max_docx_uncompressed_bytes,
            minimum=1,
            maximum=MAX_DOCX_UNCOMPRESSED_BYTES_HARD_LIMIT,
        ):
            raise ValueError("max_docx_uncompressed_bytes is outside the supported range")
        ratio = self.max_docx_compression_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 1.0 <= float(ratio) <= MAX_DOCX_COMPRESSION_RATIO_HARD_LIMIT
        ):
            raise ValueError("max_docx_compression_ratio is outside the supported range")
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "format", normalized_format)
        object.__setattr__(self, "max_docx_compression_ratio", float(ratio))

    @property
    def detected_format(self) -> str:
        if self.format != "auto":
            return self.format
        suffix = PurePosixPath(self.path).suffix.lower()
        if suffix == ".json":
            return "json"
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix in {".py", ".pyw"}:
            return "python"
        if suffix == ".docx":
            return "docx"
        if suffix in _TEXT_SUFFIXES:
            return "text"
        return "binary"


@dataclass(frozen=True)
class VerificationResult:
    artifact_id: str
    passed: bool
    detected_format: str
    checks_run: tuple[str, ...]
    errors: tuple[str, ...]
    size_bytes: int | None = None

    @property
    def message(self) -> str:
        return "verified" if self.passed else "; ".join(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactVerifier:
    """Deterministic verifier over supplied ToolResult evidence only."""

    @staticmethod
    def verify(
        spec: ArtifactSpec,
        tool_result: ToolResult | Mapping[str, Any],
        *,
        content: bytes | bytearray | memoryview | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        return verify_artifact(spec, tool_result, content=content, metadata=metadata)

    @staticmethod
    def verify_receipt(
        spec: ArtifactSpec,
        tool_result: ToolResult | Mapping[str, Any],
    ) -> VerificationResult:
        return verify_artifact_receipt(spec, tool_result)


def verify_artifact(
    spec: ArtifactSpec,
    tool_result: ToolResult | Mapping[str, Any],
    *,
    content: bytes | bytearray | memoryview | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Verify one artifact without filesystem access or command execution."""

    if not isinstance(spec, ArtifactSpec):
        raise TypeError("spec must be an ArtifactSpec")
    success, result_data = _tool_result_evidence(tool_result)
    merged, metadata_error = _merge_metadata(result_data, metadata)
    checks = ["tool_result"]
    errors: list[str] = []
    if not success:
        errors.append("tool result was unsuccessful")
    if bool(merged.get("not_executed")):
        errors.append("tool result records that execution was skipped")
    if metadata_error:
        errors.append(metadata_error)
    if errors:
        return _result(spec, checks, errors, None)

    payload, payload_error = _payload_bytes(content, merged.get("content"), spec.max_bytes)
    if payload_error:
        errors.append(payload_error)

    checks.append("exists")
    exists, exists_error = _provided_existence(merged, payload is not None)
    if exists_error:
        errors.append(exists_error)
    if exists is not True:
        errors.append("artifact existence was not proven by supplied ToolResult evidence")

    checks.extend(("nonempty", "max_size"))
    reported_size, size_error = _reported_size(merged)
    if size_error:
        errors.append(size_error)
    actual_size = len(payload) if payload is not None else reported_size
    if payload is not None and reported_size is not None and reported_size != len(payload):
        errors.append("reported artifact size does not match supplied content")
    if actual_size is None:
        errors.append("artifact size was not provided")
    elif actual_size < spec.min_bytes:
        errors.append(f"artifact is smaller than the required {spec.min_bytes} bytes")
    elif actual_size > spec.max_bytes:
        errors.append(f"artifact exceeds the {spec.max_bytes} byte limit")

    detected_format = spec.detected_format
    if detected_format in {"text", "json", "yaml", "python", "docx"}:
        checks.append("content")
        if payload is None:
            errors.append(f"{detected_format} verification requires supplied complete content")
        elif (
            bool(merged.get("source_truncated"))
            or bool(merged.get("truncated"))
            or merged.get("content_complete") is False
        ):
            errors.append("artifact content is truncated and cannot be verified")
        elif len(payload) <= spec.max_bytes:
            payload_errors, payload_checks = _verify_payload(detected_format, payload, spec)
            checks.extend(payload_checks)
            errors.extend(payload_errors)

    return _result(spec, checks, errors, actual_size)


def artifact_verification_metadata(
    spec: ArtifactSpec,
    result: VerificationResult,
    *,
    content: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    """Build a bounded receipt after complete managed content verification.

    The receipt deliberately contains no artifact body.  Runtime and Session
    consumers can therefore validate what the managed tool proved without
    opening the workspace file or persisting document contents a second time.
    """

    if not isinstance(spec, ArtifactSpec):
        raise TypeError("spec must be an ArtifactSpec")
    if not isinstance(result, VerificationResult):
        raise TypeError("result must be a VerificationResult")
    if result.artifact_id != spec.artifact_id or result.detected_format != spec.detected_format:
        raise ValueError("verification result does not match the artifact specification")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("managed artifact receipt content must be bytes")
    payload = bytes(content)
    if len(payload) > MAX_ARTIFACT_BYTES_HARD_LIMIT:
        raise ValueError("managed artifact receipt content exceeds the hard byte limit")
    if result.size_bytes != len(payload):
        raise ValueError("verification result size does not match managed artifact content")
    return {
        "schema_version": ARTIFACT_VERIFICATION_SCHEMA_VERSION,
        "artifact_id": spec.artifact_id,
        "path": spec.path,
        "format": result.detected_format,
        "passed": result.passed,
        "content_complete": True,
        "size_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "checks_run": list(result.checks_run),
        "errors": list(result.errors),
    }


def verify_artifact_receipt(
    spec: ArtifactSpec,
    tool_result: ToolResult | Mapping[str, Any],
) -> VerificationResult:
    """Validate a managed verification receipt without reading artifact bytes."""

    if not isinstance(spec, ArtifactSpec):
        raise TypeError("spec must be an ArtifactSpec")
    success, result_data = _tool_result_evidence(tool_result)
    checks = ["tool_result", "managed_receipt"]
    errors: list[str] = []
    if not success:
        errors.append("tool result was unsuccessful")
    if bool(result_data.get("not_executed")):
        errors.append("tool result records that execution was skipped")
    receipt = result_data.get(ARTIFACT_VERIFICATION_METADATA_KEY)
    if not isinstance(receipt, Mapping):
        errors.append("managed artifact verification metadata is missing")
        return _result(spec, checks, errors, None)
    if len(receipt) > MAX_METADATA_KEYS:
        errors.append(f"managed artifact verification metadata exceeds {MAX_METADATA_KEYS} keys")
        return _result(spec, checks, errors, None)

    if receipt.get("schema_version") != ARTIFACT_VERIFICATION_SCHEMA_VERSION:
        errors.append("managed artifact verification metadata has an unsupported schema")
    if receipt.get("artifact_id") != spec.artifact_id:
        errors.append("managed artifact verification metadata has the wrong artifact identity")
    try:
        receipt_path = _bounded_relative_path(receipt.get("path"))
    except ValueError:
        receipt_path = ""
        errors.append("managed artifact verification metadata has an invalid path")
    if receipt_path and receipt_path != spec.path:
        errors.append("managed artifact verification metadata path does not match the applied artifact")
    if receipt.get("format") != spec.detected_format:
        errors.append("managed artifact verification metadata has the wrong format")
    if receipt.get("passed") is not True:
        errors.append("managed artifact verification did not pass")
    if receipt.get("content_complete") is not True:
        errors.append("managed artifact verification did not inspect complete content")

    size = receipt.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        errors.append("managed artifact verification metadata has an invalid size")
        size_value: int | None = None
    else:
        size_value = size
        if size < spec.min_bytes:
            errors.append(f"artifact is smaller than the required {spec.min_bytes} bytes")
        elif size > spec.max_bytes:
            errors.append(f"artifact exceeds the {spec.max_bytes} byte limit")

    digest = receipt.get("content_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        errors.append("managed artifact verification metadata has an invalid content hash")
    receipt_errors = receipt.get("errors")
    if not isinstance(receipt_errors, list) or any(not isinstance(item, str) for item in receipt_errors):
        errors.append("managed artifact verification metadata has invalid errors")
    elif receipt_errors:
        errors.append("managed artifact verification metadata records failed checks")
    receipt_checks = receipt.get("checks_run")
    if (
        not isinstance(receipt_checks, list)
        or len(receipt_checks) > 64
        or any(not isinstance(item, str) or len(item) > 128 for item in receipt_checks)
    ):
        errors.append("managed artifact verification metadata has invalid checks")
    else:
        required = (
            _DOCX_RECEIPT_CHECKS
            if spec.detected_format == "docx"
            else {
                "tool_result",
                "exists",
                "nonempty",
                "max_size",
            }
        )
        missing = sorted(required - set(receipt_checks))
        if missing:
            errors.append("managed artifact verification metadata is missing required checks: " + ", ".join(missing))
    return _result(spec, checks, errors, size_value)


def _verify_payload(
    detected_format: str,
    payload: bytes,
    spec: ArtifactSpec,
) -> tuple[list[str], tuple[str, ...]]:
    if detected_format == "docx":
        return _verify_docx(payload, spec)
    text, error = _decode_utf8(payload)
    if error:
        return [error], ()
    assert text is not None
    if not text.strip():
        return ["artifact contains no non-whitespace text"], ()
    if detected_format == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return [f"invalid JSON syntax at line {exc.lineno}, column {exc.colno}"], ()
        except (RecursionError, ValueError):
            return ["invalid or excessively nested JSON content"], ()
    elif detected_format == "yaml":
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
            return [f"invalid YAML syntax{location}"], ()
        except (RecursionError, ValueError):
            return ["invalid or excessively nested YAML content"], ()
    elif detected_format == "python":
        try:
            ast.parse(text, filename=spec.path)
        except SyntaxError as exc:
            return [f"invalid Python syntax at line {exc.lineno or 1}, column {exc.offset or 1}"], ()
        except (RecursionError, ValueError):
            return ["invalid or excessively nested Python content"], ()
    return [], ()


def _verify_docx(payload: bytes, spec: ArtifactSpec) -> tuple[list[str], tuple[str, ...]]:
    checks: list[str] = []
    if not zipfile.is_zipfile(BytesIO(payload)):
        return ["DOCX content is not a valid ZIP container"], ()
    checks.append("docx_zip_container")
    try:
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_MEMBERS:
                return [f"DOCX contains more than {MAX_DOCX_MEMBERS} ZIP members"], tuple(checks)
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                return ["DOCX contains duplicate ZIP member names"], tuple(checks)
            unsafe = next((name for name in names if not _safe_zip_member(name)), None)
            if unsafe is not None:
                return ["DOCX contains an unsafe ZIP member path"], tuple(checks)
            encrypted = next((item for item in members if item.flag_bits & 0x1), None)
            if encrypted is not None:
                return ["DOCX contains an encrypted ZIP member"], tuple(checks)
            symlink = next((item for item in members if stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF)), None)
            if symlink is not None:
                return ["DOCX contains a symbolic-link ZIP member"], tuple(checks)
            missing = sorted(_DOCX_REQUIRED_MEMBERS - set(names))
            if missing:
                return ["DOCX is missing required package members: " + ", ".join(missing)], tuple(checks)
            checks.append("docx_package_structure")

            total_uncompressed = 0
            total_compressed = 0
            for item in members:
                if item.file_size < 0 or item.compress_size < 0:
                    return ["DOCX contains invalid ZIP member sizes"], tuple(checks)
                total_uncompressed += item.file_size
                total_compressed += item.compress_size
                if total_uncompressed > spec.max_docx_uncompressed_bytes:
                    return [f"DOCX uncompressed content exceeds {spec.max_docx_uncompressed_bytes} bytes"], tuple(
                        checks
                    )
                if item.file_size > 0:
                    if item.compress_size <= 0:
                        return ["DOCX contains a ZIP member with an unsafe compression ratio"], tuple(checks)
                    if item.file_size / item.compress_size > spec.max_docx_compression_ratio:
                        return ["DOCX contains a ZIP member with an unsafe compression ratio"], tuple(checks)
            if total_uncompressed and total_uncompressed / max(1, total_compressed) > spec.max_docx_compression_ratio:
                return ["DOCX aggregate compression ratio exceeds the configured limit"], tuple(checks)

            bad_member = archive.testzip()
            if bad_member is not None:
                return ["DOCX contains a corrupt ZIP member"], tuple(checks)
            checks.append("docx_zip_integrity")
            xml_payloads = {name: archive.read(name) for name in _DOCX_REQUIRED_MEMBERS}
    except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return ["DOCX ZIP structure could not be verified"], tuple(checks)

    roots: dict[str, ElementTree.Element] = {}
    try:
        for name, xml_payload in xml_payloads.items():
            if b"<!DOCTYPE" in xml_payload.upper() or b"<!ENTITY" in xml_payload.upper():
                return ["DOCX package XML contains a prohibited document type or entity declaration"], tuple(checks)
            roots[name] = ElementTree.fromstring(xml_payload)
    except ElementTree.ParseError:
        return ["DOCX contains invalid package XML"], tuple(checks)
    checks.append("docx_package_xml")
    document = roots["word/document.xml"]
    text = "".join(node.text or "" for node in document.iter() if node.tag == _WORD_TEXT_TAG)
    if not text.strip():
        return ["DOCX document contains no non-empty text"], tuple(checks)
    checks.append("docx_nonempty_text")
    return [], tuple(checks)


def _tool_result_evidence(tool_result: ToolResult | Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
    if isinstance(tool_result, Mapping):
        data = tool_result.get("data")
        return tool_result.get("success") is True, data if isinstance(data, Mapping) else {}
    success = getattr(tool_result, "success", None)
    data = getattr(tool_result, "data", None)
    if not isinstance(success, bool):
        raise TypeError("tool_result must provide boolean success and mapping data evidence")
    return success is True, data if isinstance(data, Mapping) else {}


def _merge_metadata(result_data: Mapping[str, Any], metadata: Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    if len(result_data) + (len(metadata) if metadata is not None else 0) > MAX_METADATA_KEYS:
        return {}, f"artifact metadata exceeds {MAX_METADATA_KEYS} keys"
    merged = dict(result_data)
    if metadata is not None:
        merged.update(metadata)
    return merged, ""


def _payload_bytes(
    explicit: bytes | bytearray | memoryview | str | None,
    embedded: Any,
    max_bytes: int,
) -> tuple[bytes | None, str]:
    selected = explicit if explicit is not None else embedded
    if selected is None:
        return None, ""
    if isinstance(selected, str):
        if len(selected) > MAX_ARTIFACT_BYTES_HARD_LIMIT:
            return None, "supplied artifact content exceeds the hard byte limit"
        encoded = selected.encode("utf-8")
    elif isinstance(selected, (bytes, bytearray, memoryview)):
        if len(selected) > MAX_ARTIFACT_BYTES_HARD_LIMIT:
            return None, "supplied artifact content exceeds the hard byte limit"
        encoded = bytes(selected)
    else:
        return None, "artifact content must be bytes or text"
    if len(encoded) > max_bytes:
        return encoded, "supplied artifact content exceeds the configured byte limit"
    return encoded, ""


def _provided_existence(metadata: Mapping[str, Any], content_exists: bool) -> tuple[bool | None, str]:
    values: list[bool] = []
    for key in ("exists", "after_exists", "restored_exists"):
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, bool):
            return None, f"artifact metadata field {key} must be boolean"
        values.append(value)
    if values and any(item != values[0] for item in values[1:]):
        return None, "artifact existence metadata is inconsistent"
    if content_exists:
        if values and values[0] is False:
            return None, "artifact content conflicts with missing-file metadata"
        return True, ""
    return (values[0], "") if values else (None, "")


def _reported_size(metadata: Mapping[str, Any]) -> tuple[int | None, str]:
    for key in ("actual_size", "size_bytes", "bytes", "size", "input_bytes"):
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, f"artifact metadata field {key} must be a non-negative integer"
        return value, ""
    return None, ""


def _decode_utf8(payload: bytes) -> tuple[str | None, str]:
    try:
        return payload.decode("utf-8-sig"), ""
    except UnicodeDecodeError:
        return None, "artifact is not valid UTF-8 text"


def _result(
    spec: ArtifactSpec,
    checks: list[str],
    errors: list[str],
    size_bytes: int | None,
) -> VerificationResult:
    unique_errors = tuple(dict.fromkeys(errors))
    return VerificationResult(
        artifact_id=spec.artifact_id,
        passed=not unique_errors,
        detected_format=spec.detected_format,
        checks_run=tuple(dict.fromkeys(checks)),
        errors=unique_errors,
        size_bytes=size_bytes,
    )


def _bounded_relative_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("artifact path must be text")
    normalized = value.strip().replace("\\", "/")
    if not normalized or len(normalized) > 1_024 or "\x00" in normalized:
        raise ValueError("artifact path must contain 1-1024 safe characters")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError("artifact path must be project-relative and cannot traverse parents")
    return path.as_posix()


def _safe_zip_member(value: str) -> bool:
    if not value or len(value) > 1_024 or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and not (path.parts and ":" in path.parts[0])


def _bounded_integer(value: Any, *, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum
