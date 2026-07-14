from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
import pty
import re
import select
import stat
import struct
import subprocess
import termios
import textwrap
import time
from pathlib import Path
from typing import Any


ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)((?:[\"'])?\b(?:[A-Za-z0-9_]*api[_-]?key|access[_-]?token|refresh[_-]?token|cookie|password|passwd|secret)"
    r"\b(?:[\"'])?\s*[=:]\s*)(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)"
)
BEARER_SECRET_RE = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=-]{12,})")
DEEPSEEK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
REPL_PROMPT_RE = re.compile(r"(?m)^[^\r\n]+:(?:new|[A-Za-z0-9_-]{3,})(?::(?:SUPER-)?YOLO)? >\s*$")
TOOL_BUDGET_MARKER = "Tool execution budget is closed. Produce the final user-facing answer"
EXCLUDED_WORD_MARKER = "EXCLUDED_INTERMEDIATE_MARKER_7F3A"
WORD_ARTIFACT_MAX_BYTES = 25_000_000
WORD_REOPENED_STDOUT_MAX_CHARS = 250_000
WORD_REOPENED_REPORT_MAX_CHARS = 260_000
WORD_DATE_VALUE_LIMIT = 100
WORD_DATE_CONTEXT_MAX_CHARS = 250_000
WORD_FIXTURE_TOPIC_GROUPS = (
    ("统一数据口径", "交付承诺", "跨部门确认"),
    ("重复录入", "缺料和返工", "一线用户"),
    ("事件驱动", "幂等校验", "失败队列"),
    ("订单答复时间", "缺料任务闭环", "设备停机"),
    ("证据链断裂", "接口重试", "人工复核"),
    ("数据质量评分", "长期产品小组", "跨工厂推广"),
)
DATE_LITERAL_RE = re.compile(r"(?<!\d)20\d{2}(?:年\s*\d{1,2}月(?:\s*\d{1,2}日)?|[-/.]\d{1,2}(?:[-/.]\d{1,2})?)(?!\d)")
GENERATED_DATE_LABEL_RE = re.compile(r"(?:生成|汇总|报告).{0,12}(?:时间|日期)|(?:时间|日期).{0,12}(?:生成|汇总|报告)")
READ_ONLY_ACTIONS = frozenset(
    {
        "document.parse",
        "git.diff",
        "git.log",
        "git.status",
        "lsp.diagnostics",
        "memory.search",
        "project.read_context",
        "template.find_files",
        "template.git_diff_staged",
        "template.list_dir",
        "template.read_file",
        "template.search_code",
    }
)
WRITE_ACTIONS = frozenset(
    {
        "browser.download",
        "file.apply",
        "file.undo",
        "git.add",
        "git.commit",
        "memory.add",
        "project.write_context",
        "template.make_dir",
    }
)
VERIFICATION_ACTIONS = frozenset(
    {
        "document.parse",
        "git.diff",
        "git.status",
        "lsp.diagnostics",
        "template.git_diff_staged",
        "template.run_tests",
    }
)
RUNNER_ERROR_NO_FINAL_PROMPT = "agent exited or stopped producing output before the final REPL prompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a timed Deep Agent PTY reliability case.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-kind", choices=("word", "text", "large"), required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def sanitize(value: str) -> str:
    value = ASSIGNMENT_SECRET_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    value = BEARER_SECRET_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    return DEEPSEEK_KEY_RE.sub("sk-[REDACTED]", value)


def terminal_screen_text(value: str) -> str:
    value = ANSI_OSC_RE.sub("", value)
    lines = [""]
    cursor = 0
    index = 0
    while index < len(value):
        if value.startswith("\r\x1b[2K", index):
            lines[-1] = ""
            cursor = 0
            index += len("\r\x1b[2K")
            continue
        char = value[index]
        if char == "\r":
            cursor = 0
        elif char == "\n":
            lines.append("")
            cursor = 0
        elif char == "\b":
            cursor = max(0, cursor - 1)
        elif char == "\x1b":
            match = ANSI_CSI_RE.match(value, index)
            if match:
                index = match.end()
                continue
        elif char not in {"\x01", "\x02"}:
            line = lines[-1]
            if cursor >= len(line):
                line += " " * (cursor - len(line)) + char
            else:
                line = line[:cursor] + char + line[cursor + 1 :]
            lines[-1] = line
            cursor += 1
        index += 1
    return "\n".join(lines)


def extract_screen_answer(screen: str) -> str:
    marker = "\nassistant\n"
    if marker not in screen:
        return ""
    tail = screen.rsplit(marker, 1)[1]
    prompt_match = REPL_PROMPT_RE.search(tail)
    if prompt_match:
        tail = tail[: prompt_match.start()]
    return tail.strip()


def repl_prompt_count(screen: str) -> int:
    return len(REPL_PROMPT_RE.findall(screen))


def has_final_repl_prompt(screen: str) -> bool:
    marker = "\nassistant\n"
    return marker in screen and REPL_PROMPT_RE.search(screen.rsplit(marker, 1)[1]) is not None


def extract_thinking_sections(screen: str) -> list[str]:
    sections: list[str] = []
    lines = screen.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "● DeepSeek Thinking":
            index += 1
            continue
        index += 1
        content: list[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped == "● DeepSeek Thinking" or stripped == "assistant" or REPL_PROMPT_RE.fullmatch(stripped):
                break
            content.append(lines[index])
            index += 1
        section = textwrap.dedent("\n".join(content)).strip()
        if section:
            sections.append(section)
    return sections


def _load_latest_session(workspace: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    session_dir = workspace / ".project-agent" / "sessions"
    session_files = sorted(session_dir.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    if not session_files:
        return {}, [], "", ""
    path = session_files[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {}, [], path.stem, f"could not parse Session JSON: {exc}"
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    return state, [item for item in messages if isinstance(item, dict)], str(state.get("session_id") or path.stem), ""


def _tool_name(item: dict[str, Any]) -> str:
    request = item.get("request") if isinstance(item.get("request"), dict) else {}
    return f"{request.get('tool', '?')}.{request.get('action', '?')}"


def _current_turn_tools(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    turn = max(1, int(state.get("turn") or 1))
    all_tools = state.get("tool_calls") if isinstance(state.get("tool_calls"), list) else []
    return turn, [item for item in all_tools if isinstance(item, dict) and int(item.get("turn") or 1) == turn]


def _max_consecutive_read_only_rounds(tool_calls: list[dict[str, Any]]) -> int:
    by_round: dict[int, list[str]] = {}
    for item in tool_calls:
        by_round.setdefault(int(item.get("round") or 0), []).append(_tool_name(item))
    longest = current = 0
    previous_round: int | None = None
    for round_number in sorted(by_round):
        read_only = bool(by_round[round_number]) and all(name in READ_ONLY_ACTIONS for name in by_round[round_number])
        consecutive = previous_round is None or round_number == previous_round + 1
        current = current + 1 if read_only and consecutive else 1 if read_only else 0
        longest = max(longest, current)
        previous_round = round_number
    return longest


def _first_round(tool_calls: list[dict[str, Any]], names: frozenset[str], *, after: int | None = None) -> int | None:
    rounds = [
        int(item.get("round") or 0)
        for item in tool_calls
        if _tool_name(item) in names
        and (after is None or int(item.get("round") or 0) >= after)
        and bool((item.get("result") or {}).get("success"))
    ]
    return min(rounds) if rounds else None


def _tool_result(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("result") if isinstance(item.get("result"), dict) else {}


def _recorded_paths(item: dict[str, Any]) -> list[str]:
    request = item.get("request") if isinstance(item.get("request"), dict) else {}
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    result = _tool_result(item)
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    paths: list[str] = []
    for value in (args.get("path"), data.get("path")):
        path = str(value or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _final_applied_docx(tool_calls: list[dict[str, Any]]) -> tuple[int, int, str, dict[str, Any]] | None:
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for index, item in enumerate(tool_calls):
        if _tool_name(item) != "file.apply" or not bool(_tool_result(item).get("success")):
            continue
        paths = [path for path in _recorded_paths(item) if path.lower().endswith(".docx")]
        if paths:
            candidates.append((index, int(item.get("round") or 0), paths[-1], item))
    return max(candidates, key=lambda value: (value[1], value[0])) if candidates else None


def _matching_record(
    tool_calls: list[dict[str, Any]],
    *,
    name: str,
    artifact_path: str,
    before_index: int | None = None,
    after_index: int | None = None,
) -> tuple[int, dict[str, Any]] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(tool_calls):
        if before_index is not None and index >= before_index:
            continue
        if after_index is not None and index <= after_index:
            continue
        if _tool_name(item) != name or not bool(_tool_result(item).get("success")):
            continue
        if any(_same_path(path, artifact_path) for path in _recorded_paths(item)):
            candidates.append((index, item))
    return max(candidates, key=lambda value: (int(value[1].get("round") or 0), value[0])) if candidates else None


def _word_artifact_trace(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    applied = _final_applied_docx(tool_calls)
    if applied is None:
        return {"artifact_path": "", "apply_index": None, "render": None, "reopened": None}
    apply_index, apply_round, artifact_path, apply_item = applied
    render = _matching_record(
        tool_calls,
        name="document.render_docx",
        artifact_path=artifact_path,
        before_index=apply_index,
    )
    reopened = _matching_record(
        tool_calls,
        name="document.parse",
        artifact_path=artifact_path,
        after_index=apply_index,
    )
    return {
        "artifact_path": artifact_path,
        "apply_index": apply_index,
        "apply_round": apply_round,
        "apply": apply_item,
        "render": render[1] if render else None,
        "reopened": reopened[1] if reopened else None,
    }


def _parsed_document_body(stdout: str) -> str:
    value = stdout[:WORD_REOPENED_STDOUT_MAX_CHARS].replace("\r\n", "\n")
    divider = re.search(r"(?m)^---\s*$", value)
    if divider:
        return value[divider.end() :].strip()
    lines = [
        line
        for line in value.splitlines()
        if line.strip() != "# Parsed Document" and not line.lstrip().startswith("- Source:")
    ]
    return "\n".join(lines).strip()


def _date_key(value: str) -> tuple[int, ...] | None:
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    if len(numbers) < 2:
        return None
    year, month = numbers[:2]
    if not (2000 <= year <= 2099 and 1 <= month <= 12):
        return None
    if len(numbers) >= 3:
        day = numbers[2]
        if not 1 <= day <= 31:
            return None
        return year, month, day
    return year, month


def _date_keys_from_text(value: str) -> set[tuple[int, ...]]:
    return {key for item in DATE_LITERAL_RE.findall(value) if (key := _date_key(item)) is not None}


def _metadata_date_values(value: Any) -> set[str]:
    items = value if isinstance(value, list) else [value]
    return {text for item in items[:WORD_DATE_VALUE_LIMIT] if (text := str(item or "").strip()[:100])}


def _generated_dates_from_text(value: str) -> set[str]:
    dates: set[str] = set()
    bounded = value[-WORD_DATE_CONTEXT_MAX_CHARS:]
    for line in bounded.splitlines():
        if GENERATED_DATE_LABEL_RE.search(line):
            dates.update(DATE_LITERAL_RE.findall(line))
            if len(dates) >= WORD_DATE_VALUE_LIMIT:
                break
    return set(sorted(dates)[:WORD_DATE_VALUE_LIMIT])


def _word_generated_dates(trace: dict[str, Any], reopened_body: str) -> set[str]:
    values = _generated_dates_from_text(reopened_body)
    render = trace.get("render") if isinstance(trace.get("render"), dict) else {}
    result = _tool_result(render)
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    values.update(_metadata_date_values(data.get("generated_metadata_dates", [])))
    request = render.get("request") if isinstance(render.get("request"), dict) else {}
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    values.update(_generated_dates_from_text(str(args.get("markdown") or "")))
    return set(sorted(values)[:WORD_DATE_VALUE_LIMIT])


def _word_allowed_date_keys(
    state: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    trace: dict[str, Any],
) -> set[tuple[int, ...]]:
    request_text = "\n".join(
        str(state.get(key) or "")[:WORD_DATE_CONTEXT_MAX_CHARS] for key in ("objective", "user_request")
    )
    allowed = _date_keys_from_text(request_text)
    apply_index = trace.get("apply_index")
    if not isinstance(apply_index, int):
        return allowed
    artifact_path = str(trace.get("artifact_path") or "")
    allowed_sources = {"document.parse", "ocr.parse", "template.read_file"}
    for index, item in enumerate(tool_calls):
        if index >= apply_index:
            break
        if _tool_name(item) not in allowed_sources or not bool(_tool_result(item).get("success")):
            continue
        if artifact_path and any(_same_path(path, artifact_path) for path in _recorded_paths(item)):
            continue
        result = _tool_result(item)
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        allowed.update(_date_keys_from_text(str(data.get("date_literals") or "")[:WORD_DATE_CONTEXT_MAX_CHARS]))
        allowed.update(_date_keys_from_text(str(result.get("stdout") or "")[:WORD_DATE_CONTEXT_MAX_CHARS]))
    return allowed


def _successful_tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [_tool_name(item) for item in tool_calls if bool((item.get("result") or {}).get("success"))]


def _is_validation_call(item: dict[str, Any]) -> bool:
    # This research snapshot is intentionally incomplete and its repository
    # typecheck can fail on missing unpublished modules.  The acceptance gate
    # measures whether validation was actually executed and reported; it must
    # not rewrite an honest failing baseline as "validation was never run".
    name = _tool_name(item)
    if name in {"lsp.diagnostics", "template.run_tests"}:
        return True
    if name != "shell.run":
        return False
    request = item.get("request") if isinstance(item.get("request"), dict) else {}
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    command = str(args.get("command") or "").lower()
    executable = re.search(r"(?:^|\s)(?:bun|npm|pnpm|yarn|npx|tsc|pytest|ruff|eslint)(?:\s|$)", command)
    return bool(executable) and any(
        marker in command for marker in ("test", "typecheck", "check", "lint", "build", "tsc")
    )


def _reports_at_least_eight_advantages(final_answer: str) -> bool:
    if re.search(r"(?:^|\n)\s*(?:8[.、)]|八[、.])", final_answer):
        return True
    if re.search(r"(?:8|八)\s*(?:个|项|点)[^\n]{0,20}优点", final_answer):
        return True
    numbered_items = re.findall(r"(?m)^\s*(?:[1-9]|[一二三四五六七八九])[.、)]\s+", final_answer)
    bullet_items = re.findall(r"(?m)^\s*[-*]\s+", final_answer)
    return "优点" in final_answer and max(len(numbered_items), len(bullet_items)) >= 8


def _same_path(left: str, right: str) -> bool:
    left = left.replace("\\", "/").rstrip("/")
    right = right.replace("\\", "/").rstrip("/")
    return bool(left and right) and (
        left == right or left.endswith("/" + right.lstrip("/")) or right.endswith("/" + left.lstrip("/"))
    )


def semantic_checks(
    case_kind: str,
    state: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    final_answer: str,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "session_completed": str(state.get("status") or "") == "completed",
        "final_answer_nonempty": bool(final_answer.strip()),
    }
    names = _successful_tool_names(tool_calls)
    if case_kind == "word":
        trace = _word_artifact_trace(tool_calls)
        artifact_path = str(trace.get("artifact_path") or "")
        apply_index = trace.get("apply_index")
        reopened = trace.get("reopened") if isinstance(trace.get("reopened"), dict) else {}
        reopened_result = _tool_result(reopened)
        reopened_stdout = str(reopened_result.get("stdout") or "")[:WORD_REOPENED_STDOUT_MAX_CHARS]
        reopened_body = _parsed_document_body(reopened_stdout)
        source_paths = {
            Path(path.replace("\\", "/")).name.lower()
            for index, item in enumerate(tool_calls)
            if isinstance(apply_index, int) and index < apply_index
            if _tool_name(item) == "document.parse" and bool(_tool_result(item).get("success"))
            for path in _recorded_paths(item)[:1]
            if path.lower().endswith(".docx") and not _same_path(path, artifact_path)
        }
        generated_dates = _word_generated_dates(trace, reopened_body)
        generated_date_keys = {key for value in generated_dates if (key := _date_key(value)) is not None}
        generated_dates_valid = all(_date_key(value) is not None for value in generated_dates)
        allowed_date_keys = _word_allowed_date_keys(state, tool_calls, trace)
        checks.update(
            {
                "parsed_at_least_six_source_documents": len(source_paths) >= 6,
                "rendered_docx_preview": isinstance(trace.get("render"), dict),
                "applied_docx_artifact": bool(artifact_path),
                "reopened_applied_docx": bool(reopened),
                "reopened_body_nonempty": bool(reopened_body),
                "excluded_intermediate_marker_absent": EXCLUDED_WORD_MARKER not in reopened_stdout,
                "covers_all_six_fixture_topics": all(
                    any(topic in reopened_body for topic in alternatives) for alternatives in WORD_FIXTURE_TOPIC_GROUPS
                ),
                "generated_metadata_dates_supported": generated_dates_valid
                and generated_date_keys.issubset(allowed_date_keys),
            }
        )
    elif case_kind == "text":
        required_topics = ("项目目标", "用户发现", "功能范围", "安全要求", "实施计划", "验收指标", "风险")
        checks.update(
            {
                "read_source_material": any(name in {"document.parse", "template.read_file"} for name in names),
                "no_managed_write_tools": not any(
                    name in WRITE_ACTIONS or name == "document.render_docx" for name in names
                ),
                "covers_requested_topics": sum(topic in final_answer for topic in required_topics) >= 5,
            }
        )
    else:
        plan = state.get("plan") if isinstance(state.get("plan"), list) else []
        completed_steps = [item for item in plan if isinstance(item, dict) and item.get("status") == "completed"]
        checks.update(
            {
                "source_evidence_present": sum(token in final_answer for token in ("src/", ".ts", ".tsx", "README"))
                >= 2,
                "reports_at_least_eight_advantages": _reports_at_least_eight_advantages(final_answer),
                "ran_static_or_test_validation": any(_is_validation_call(item) for item in tool_calls),
                "plan_advanced": bool(completed_steps),
                "bug_or_limitation_reported": any(
                    token in final_answer.lower() for token in ("bug", "缺陷", "未找到", "没有充分证据", "验证限制")
                ),
                "verification_result_reported": any(token in final_answer for token in ("验证", "检查", "测试")),
            }
        )
    return {"passed": all(checks.values()), "checks": checks}


def _resolve_workspace_artifact(workspace: Path, recorded_path: str) -> tuple[Path, str]:
    root = workspace.resolve(strict=True)
    candidate = Path(recorded_path.replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("final artifact resolves outside the workspace") from exc
    if not relative.parts or resolved.suffix.lower() != ".docx":
        raise ValueError("final artifact is not a workspace .docx file")
    return resolved, relative.as_posix()


def _copy_bounded_regular_file(source: Path, destination: Path) -> int:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, source_flags)
    temporary = destination.with_name(f".{destination.name}.tmp")
    destination_fd: int | None = None
    copied = 0
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("final artifact is not a regular file")
        if source_stat.st_size <= 0:
            raise ValueError("final artifact is empty")
        if source_stat.st_size > WORD_ARTIFACT_MAX_BYTES:
            raise ValueError(f"final artifact exceeds {WORD_ARTIFACT_MAX_BYTES} bytes")
        destination_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_fd, 65_536)
            if not chunk:
                break
            copied += len(chunk)
            if copied > WORD_ARTIFACT_MAX_BYTES:
                raise ValueError(f"final artifact exceeds {WORD_ARTIFACT_MAX_BYTES} bytes")
            pending = memoryview(chunk)
            while pending:
                written = os.write(destination_fd, pending)
                if written <= 0:
                    raise OSError("could not write artifact evidence")
                pending = pending[written:]
        if copied != source_stat.st_size:
            raise ValueError("final artifact changed while evidence was copied")
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
        os.replace(temporary, destination)
        return copied
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        if temporary.exists():
            temporary.unlink()


def _write_word_artifact_evidence(
    *,
    workspace: Path,
    output: Path,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    trace = _word_artifact_trace(tool_calls)
    artifact_path = str(trace.get("artifact_path") or "")
    reopened = trace.get("reopened") if isinstance(trace.get("reopened"), dict) else {}
    reopened_stdout = str(_tool_result(reopened).get("stdout") or "")
    result: dict[str, Any] = {
        "passed": False,
        "artifact_copied": False,
        "report_written": False,
        "artifact_file": "",
        "reopened_report_file": "re-opened-artifact.md",
        "workspace_relative_source": "",
        "artifact_bytes": 0,
        "reopened_stdout_chars": len(reopened_stdout),
        "reopened_stdout_truncated": len(reopened_stdout) > WORD_REOPENED_STDOUT_MAX_CHARS,
        "error": "",
    }
    try:
        if not artifact_path:
            raise ValueError("no successful applied .docx was recorded in the current turn")
        if not reopened:
            raise ValueError("no matching document.parse after the final .docx apply was recorded")
        source, relative = _resolve_workspace_artifact(workspace, artifact_path)
        destination = output.resolve(strict=True) / "final-artifact.docx"
        copied = _copy_bounded_regular_file(source, destination)
        result.update(
            {
                "artifact_copied": True,
                "artifact_file": destination.name,
                "workspace_relative_source": sanitize(relative)[:1000],
                "artifact_bytes": copied,
            }
        )
    except (OSError, ValueError) as exc:
        result["error"] = sanitize(f"{type(exc).__name__}: {exc}")[:1000]

    status = "passed" if result["artifact_copied"] else "failed"
    source_label = str(result["workspace_relative_source"] or "unavailable").replace("`", "'")
    error_line = f"\n- Error: {result['error']}" if result["error"] else ""
    parsed_output = sanitize(reopened_stdout[:WORD_REOPENED_STDOUT_MAX_CHARS])
    report = (
        "# Re-opened Word artifact evidence\n\n"
        f"- Evidence status: {status}\n"
        f"- Workspace-relative source: `{source_label}`\n"
        f"- Copied artifact: `{result['artifact_file'] or 'not copied'}`\n"
        f"- Artifact bytes: {result['artifact_bytes']}\n"
        f"- document.parse stdout characters: {result['reopened_stdout_chars']}\n"
        f"- document.parse stdout truncated: {str(result['reopened_stdout_truncated']).lower()}"
        f"{error_line}\n\n"
        "## document.parse stdout\n\n"
        f"{parsed_output or '[no re-opened document body was recorded]'}\n"
    )
    if len(report) > WORD_REOPENED_REPORT_MAX_CHARS:
        report = report[: WORD_REOPENED_REPORT_MAX_CHARS - 40] + "\n\n[report truncated by runner]\n"
    report_path = output.resolve(strict=True) / "re-opened-artifact.md"
    try:
        report_path.write_text(report, encoding="utf-8")
        report_path.chmod(0o600)
        result["report_written"] = True
    except OSError as exc:
        result["error"] = sanitize(f"{type(exc).__name__}: {exc}")[:1000]
    result["passed"] = bool(result["artifact_copied"] and result["report_written"])
    return result


def build_metrics(
    *,
    case_kind: str,
    workspace: Path,
    state: dict[str, Any],
    messages: list[dict[str, Any]],
    session_id: str,
    session_error: str,
    screen: str,
    raw_text: str,
    screen_answer: str,
    timing: dict[str, float | None],
    process_returncode: int | None,
    runner_error: str,
    timed_out: bool,
    final_prompt_seen: bool,
    word_artifact_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_answer = str(state.get("final_answer") or "").strip()
    turn, tool_calls = _current_turn_tools(state)
    counts = Counter(_tool_name(item) for item in tool_calls)
    thinking_sections = extract_thinking_sections(screen)
    thinking_chars = sum(len(item) for item in thinking_sections)
    dot_like = sum(bool(re.fullmatch(r"[.。…·\s]+", item)) for item in thinking_sections)
    main_loop_requests = max(
        0,
        int(
            state.get("main_loop_model_request_count")
            if "main_loop_model_request_count" in state
            else state.get("round") or 0
        ),
    )
    context_compaction_requests = max(0, int(state.get("context_compaction_model_request_count") or 0))
    final_synthesis_requests = max(
        0,
        int(
            state.get("final_synthesis_model_request_count")
            if "final_synthesis_model_request_count" in state
            else sum(
                TOOL_BUDGET_MARKER in str(item.get("content") or "")
                for item in messages
                if item.get("role") == "system"
            )
        ),
    )
    logical_requests = max(
        0,
        int(
            state.get("model_request_count")
            if "model_request_count" in state
            else main_loop_requests + context_compaction_requests + final_synthesis_requests
        ),
    )
    write_round = _first_round(tool_calls, WRITE_ACTIONS)
    verification_round = _first_round(tool_calls, VERIFICATION_ACTIONS, after=write_round) if write_round else None
    answer_matches_screen = bool(final_answer) and final_answer == screen_answer
    semantic = semantic_checks(case_kind, state, tool_calls, final_answer)
    if case_kind == "word":
        evidence_saved = bool(word_artifact_evidence and word_artifact_evidence.get("passed"))
        semantic["checks"]["artifact_evidence_saved"] = evidence_saved
        semantic["passed"] = bool(semantic["passed"] and evidence_saved)
    passed = bool(
        semantic["passed"]
        and final_prompt_seen
        and answer_matches_screen
        and not runner_error
        and not timed_out
        and process_returncode == 0
    )

    submitted_at = timing.get("submitted_at")
    finished_at = timing.get("finished_at") or time.monotonic()

    def duration(end_key: str, start_key: str) -> float | None:
        end = timing.get(end_key)
        start = timing.get(start_key)
        return round(end - start, 3) if end is not None and start is not None else None

    return {
        "case_kind": case_kind,
        "workspace": str(workspace),
        "session_id": session_id,
        "session_load_error": session_error,
        "status": str(state.get("status") or "unknown"),
        "turn": turn,
        "model": str((state.get("model_route") or {}).get("model") or "unknown"),
        "model_tier": str((state.get("model_route") or {}).get("tier") or "unknown"),
        "task_mode": str((state.get("task_route") or {}).get("mode") or "unknown"),
        "main_loop_model_request_count": main_loop_requests,
        "context_compaction_model_request_count": context_compaction_requests,
        "final_synthesis_model_request_count": final_synthesis_requests,
        "logical_model_request_count": logical_requests,
        "http_attempt_count": int((state.get("model_metrics") or {}).get("http_attempt_count") or 0),
        "model_usage": {
            key: int((state.get("model_metrics") or {}).get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "tool_call_count": len(tool_calls),
        "tool_call_counts": dict(sorted(counts.items())),
        "successful_tool_call_count": sum(bool((item.get("result") or {}).get("success")) for item in tool_calls),
        "failed_tool_call_count": sum(not bool((item.get("result") or {}).get("success")) for item in tool_calls),
        "max_consecutive_read_only_rounds": _max_consecutive_read_only_rounds(tool_calls),
        "first_write_round": write_round,
        "first_verification_round_after_write": verification_round,
        "tool_calls": [
            {
                "round": int(item.get("round") or 0),
                "name": _tool_name(item),
                "success": bool((item.get("result") or {}).get("success")),
                "duration_ms": int((item.get("result") or {}).get("duration_ms") or 0),
            }
            for item in tool_calls
        ],
        "time_to_prompt_seconds": duration("prompt_seen_at", "started_at"),
        "time_to_first_thinking_seconds": duration("first_thinking_at", "submitted_at"),
        "time_to_first_answer_seconds": duration("first_assistant_at", "submitted_at"),
        "time_to_completed_answer_seconds": duration("final_prompt_at", "submitted_at"),
        "total_task_seconds": round(finished_at - submitted_at, 3) if submitted_at is not None else None,
        "wall_seconds": duration("finished_at", "started_at"),
        "thinking_section_count": len(thinking_sections),
        "thinking_chars": thinking_chars,
        "thinking_first_excerpt": sanitize(thinking_sections[0][:500]) if thinking_sections else "",
        "thinking_dot_like_section_count": dot_like,
        "screen_answer_chars": len(screen_answer),
        "session_final_answer_chars": len(final_answer),
        "screen_answer_matches_session": answer_matches_screen,
        "final_prompt_seen": final_prompt_seen,
        "timed_out": timed_out,
        "runner_error": sanitize(runner_error),
        "process_returncode": process_returncode,
        "semantic_gate": semantic,
        "word_artifact_evidence": word_artifact_evidence or {},
        "passed": passed,
        "raw_output_chars": len(raw_text),
    }


def _prepare_output_dir(path: Path) -> Path:
    output = path.resolve()
    if output.exists():
        raise SystemExit(f"output directory must be new and must not already exist: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _set_pty_size(fd: int, rows: int = 40, columns: int = 120) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _agent_environment() -> dict[str, str]:
    return {
        **os.environ,
        "TERM": "xterm-256color",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "COLUMNS": "120",
        "LINES": "40",
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = _prepare_output_dir(args.output)
    prompt = " ".join(args.prompt_file.read_text(encoding="utf-8").split())
    if not prompt:
        raise SystemExit("prompt file is empty")
    workspace.mkdir(parents=True, exist_ok=True)
    agent_dir = workspace / ".project-agent"
    if agent_dir.exists():
        raise SystemExit(f"workspace already contains Agent state: {agent_dir}")

    raw = bytearray()
    timing: dict[str, float | None] = {
        "started_at": time.monotonic(),
        "prompt_seen_at": None,
        "submitted_at": None,
        "first_thinking_at": None,
        "first_assistant_at": None,
        "final_prompt_at": None,
        "finished_at": None,
    }
    process: subprocess.Popen[bytes] | None = None
    master: int | None = None
    runner_error = ""
    timed_out = False
    final_prompt_seen = False

    try:
        master, slave = pty.openpty()
        _set_pty_size(slave)
        process = subprocess.Popen(
            ["agent", "--yolo"],
            cwd=workspace,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=_agent_environment(),
            close_fds=True,
        )
        os.close(slave)
        deadline = timing["started_at"] + args.timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 65_536)
                except OSError as exc:
                    if process.poll() is not None:
                        break
                    raise exc
                if not chunk:
                    break
                raw.extend(chunk)
                now = time.monotonic()
                screen = terminal_screen_text(raw.decode("utf-8", errors="replace"))
                if timing["prompt_seen_at"] is None and repl_prompt_count(screen) >= 1:
                    timing["prompt_seen_at"] = now
                    os.write(master, prompt.encode("utf-8") + b"\r")
                    timing["submitted_at"] = time.monotonic()
                if timing["submitted_at"] is not None and timing["first_thinking_at"] is None:
                    if "● DeepSeek Thinking" in screen:
                        timing["first_thinking_at"] = now
                if timing["submitted_at"] is not None and timing["first_assistant_at"] is None:
                    if "\nassistant\n" in screen:
                        timing["first_assistant_at"] = now
                if timing["submitted_at"] is not None and has_final_repl_prompt(screen):
                    final_prompt_seen = True
                    timing["final_prompt_at"] = now
                    break
            if process.poll() is not None:
                break
        else:
            timed_out = True
            runner_error = f"agent did not return to a new REPL prompt within {args.timeout:.0f}s"

        if not final_prompt_seen and not timed_out and not runner_error:
            runner_error = RUNNER_ERROR_NO_FINAL_PROMPT
        if final_prompt_seen and master is not None and process.poll() is None:
            os.write(master, b"/exit\r")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
    except Exception as exc:  # Evidence must still be written for every runner failure.
        runner_error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if master is not None:
            os.close(master)
        timing["finished_at"] = time.monotonic()

    raw_text = sanitize(raw.decode("utf-8", errors="replace"))
    screen = sanitize(terminal_screen_text(raw_text))
    screen_answer = extract_screen_answer(screen)
    state, messages, session_id, session_error = _load_latest_session(workspace)
    final_answer = sanitize(str(state.get("final_answer") or "").strip())
    state = dict(state)
    state["final_answer"] = final_answer
    _, current_turn_tools = _current_turn_tools(state)
    word_artifact_evidence = (
        _write_word_artifact_evidence(
            workspace=workspace,
            output=output,
            tool_calls=current_turn_tools,
        )
        if args.case_kind == "word"
        else None
    )
    metrics = build_metrics(
        case_kind=args.case_kind,
        workspace=workspace,
        state=state,
        messages=messages,
        session_id=session_id,
        session_error=session_error,
        screen=screen,
        raw_text=raw_text,
        screen_answer=screen_answer,
        timing=timing,
        process_returncode=process.returncode if process is not None else None,
        runner_error=runner_error,
        timed_out=timed_out,
        final_prompt_seen=final_prompt_seen,
        word_artifact_evidence=word_artifact_evidence,
    )
    (output / "pty-raw.txt").write_text(raw_text, encoding="utf-8")
    (output / "pty-screen.txt").write_text(screen, encoding="utf-8")
    (output / "final-answer.md").write_text(final_answer + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
