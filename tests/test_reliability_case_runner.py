from __future__ import annotations

import importlib.util
import fcntl
import os
from pathlib import Path
import pty
import struct
import sys
import termios

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reliability_case_runner.py"
SPEC = importlib.util.spec_from_file_location("reliability_case_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def tool_call(
    turn: int,
    round_number: int,
    name: str,
    *,
    success: bool = True,
    args: dict[str, object] | None = None,
    data: dict[str, object] | None = None,
    stdout: str = "",
) -> dict[str, object]:
    tool, action = name.split(".", 1)
    return {
        "turn": turn,
        "round": round_number,
        "request": {"tool": tool, "action": action, "args": args or {}},
        "result": {"success": success, "duration_ms": 5, "data": data or {}, "stdout": stdout},
    }


def timing() -> dict[str, float | None]:
    return {
        "started_at": 10.0,
        "prompt_seen_at": 10.5,
        "submitted_at": 11.0,
        "first_thinking_at": 12.0,
        "first_assistant_at": 15.0,
        "final_prompt_at": 16.0,
        "finished_at": 16.5,
    }


def word_summary_body(*, suffix: str = "") -> str:
    return (
        "项目通过统一数据口径降低交付承诺风险；一线用户希望减少重复录入；"
        "技术方案采用事件驱动和幂等校验；实施后订单答复时间明显缩短；"
        "主要风险是系统证据链断裂，因此保留人工复核；后续建立数据质量评分并推进跨工厂推广。"
        f"{suffix}"
    )


def valid_word_tools(*, workspace: Path | None = None, reopened_suffix: str = "") -> list[dict[str, object]]:
    artifact = workspace / "summary.docx" if workspace is not None else Path("summary.docx")
    artifact_path = str(artifact)
    return [
        *[
            tool_call(
                1,
                1,
                "document.parse",
                args={"path": f"{index:02d}.docx"},
                stdout=f"# Parsed Document\n\n---\n\nsource {index}",
            )
            for index in range(1, 7)
        ],
        tool_call(
            1,
            2,
            "document.render_docx",
            args={"path": artifact_path, "markdown": word_summary_body(suffix=reopened_suffix)},
            data={"path": "summary.docx", "generated_metadata_dates": []},
        ),
        tool_call(1, 3, "file.apply", data={"path": "summary.docx"}),
        tool_call(
            1,
            4,
            "document.parse",
            args={"path": artifact_path},
            stdout=f"# Parsed Document\n\n- Source: `{artifact_path}`\n\n---\n\n"
            + word_summary_body(suffix=reopened_suffix),
        ),
    ]


def test_sanitize_redacts_obvious_credentials_without_corrupting_source_terms() -> None:
    value = "\n".join(
        [
            "DEEPSEEK_API_KEY=sk-1234567890abcdef",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            '"password": "correct-horse-battery-staple"',
            "token: string",
            "task-notification",
            "short sk-demo",
        ]
    )

    sanitized = runner.sanitize(value)

    assert "sk-1234567890abcdef" not in sanitized
    assert "Bearer [REDACTED]" in sanitized
    assert '"password": [REDACTED]' in sanitized
    assert "token: string" in sanitized
    assert "task-notification" in sanitized
    assert "short sk-demo" in sanitized


def test_terminal_parsing_requires_prompt_after_assistant_and_extracts_thinking() -> None:
    screen = "\n".join(
        [
            "project:new:YOLO > request",
            "  ● DeepSeek Thinking",
            "    inspect bounded chunks",
            "assistant",
            "complete answer",
        ]
    )
    assert not runner.has_final_repl_prompt(screen)

    completed = screen + "\n\nproject:abc12345:YOLO > "
    assert runner.has_final_repl_prompt(completed)
    assert runner.extract_screen_answer(completed) == "complete answer"
    assert runner.extract_thinking_sections(completed) == ["inspect bounded chunks"]


def test_output_directory_must_be_new(tmp_path: Path) -> None:
    created = runner._prepare_output_dir(tmp_path / "new")
    assert created.is_dir()

    with pytest.raises(SystemExit, match="must not already exist"):
        runner._prepare_output_dir(created)


def test_agent_environment_and_pty_dimensions_are_fixed() -> None:
    env = runner._agent_environment()
    assert env["LANG"] == "C.UTF-8"
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["COLUMNS"] == "120"
    assert env["LINES"] == "40"

    master, slave = pty.openpty()
    try:
        runner._set_pty_size(slave)
        rows, columns, _, _ = struct.unpack("HHHH", fcntl.ioctl(slave, termios.TIOCGWINSZ, b"\0" * 8))
        assert (rows, columns) == (40, 120)
    finally:
        os.close(master)
        os.close(slave)


def test_metrics_use_current_turn_and_count_final_synthesis() -> None:
    previous_turn = tool_call(1, 9, "file.apply", data={"path": "old.txt"})
    current_tools = [
        tool_call(2, 1, "template.read_file"),
        tool_call(2, 2, "template.search_code"),
        tool_call(2, 3, "file.apply", data={"path": "src/fix.ts"}),
        tool_call(2, 4, "template.run_tests"),
    ]
    final = (
        "项目包含 README 与 src/query.ts。\n"
        "1. 优点\n2. 优点\n3. 优点\n4. 优点\n5. 优点\n6. 优点\n7. 优点\n8. 优点\n"
        "Bug 缺陷已修复，测试和验证通过。"
    )
    state = {
        "status": "completed",
        "turn": 2,
        "round": 4,
        "final_answer": final,
        "tool_calls": [previous_turn, *current_tools],
        "plan": [{"id": "inspect", "status": "completed"}],
        "model_route": {"model": "deepseek-test", "tier": "deep"},
        "task_route": {"mode": "deep"},
    }
    screen = (
        "project:new:YOLO > task\n"
        "  ● DeepSeek Thinking\n"
        "    inspect bounded chunks\n"
        "  ● DeepSeek Thinking\n"
        "    .\n"
        f"assistant\n{final}\n\nproject:session1:YOLO > "
    )
    messages = [{"role": "system", "content": runner.TOOL_BUDGET_MARKER + ". Do not call tools."}]

    metrics = runner.build_metrics(
        case_kind="large",
        workspace=Path("/tmp/project"),
        state=state,
        messages=messages,
        session_id="session1",
        session_error="",
        screen=screen,
        raw_text=screen,
        screen_answer=final,
        timing=timing(),
        process_returncode=0,
        runner_error="",
        timed_out=False,
        final_prompt_seen=True,
    )

    assert metrics["main_loop_model_request_count"] == 4
    assert metrics["context_compaction_model_request_count"] == 0
    assert metrics["final_synthesis_model_request_count"] == 1
    assert metrics["logical_model_request_count"] == 5
    assert metrics["tool_call_count"] == 4
    assert metrics["tool_call_counts"] == {
        "file.apply": 1,
        "template.read_file": 1,
        "template.run_tests": 1,
        "template.search_code": 1,
    }
    assert metrics["max_consecutive_read_only_rounds"] == 2
    assert metrics["first_write_round"] == 3
    assert metrics["first_verification_round_after_write"] == 4
    assert metrics["thinking_chars"] == len("inspect bounded chunks") + 1
    assert metrics["thinking_first_excerpt"] == "inspect bounded chunks"
    assert metrics["thinking_dot_like_section_count"] == 1
    assert metrics["screen_answer_matches_session"] is True
    assert metrics["semantic_gate"]["passed"] is True
    assert metrics["passed"] is True


def test_large_gate_counts_an_honestly_reported_failed_validation_attempt() -> None:
    failed_validation = tool_call(1, 2, "template.run_tests", success=False)
    final = (
        "README 与 src/query.ts 已核对。\n"
        "1. 优点\n2. 优点\n3. 优点\n4. 优点\n5. 优点\n6. 优点\n7. 优点\n8. 优点\n"
        "未找到可独立修复的新 Bug；验证限制：typecheck 已执行但因快照缺失内部模块而未通过，检查结果已报告。"
    )
    state = {
        "status": "completed",
        "plan": [{"id": "verify", "status": "completed"}],
    }

    result = runner.semantic_checks("large", state, [failed_validation], final)

    assert result["checks"]["ran_static_or_test_validation"] is True
    assert result["checks"]["bug_or_limitation_reported"] is True
    assert result["passed"] is True


def test_word_gate_requires_render_apply_and_reopen_same_docx() -> None:
    tools = valid_word_tools()
    state = {"status": "completed"}

    passed = runner.semantic_checks("word", state, tools, "完成并验证 summary.docx")
    failed = runner.semantic_checks("word", state, tools[:-1], "完成 summary.docx")

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["reopened_applied_docx"] is False


def test_word_gate_rejects_old_technical_false_positive_with_unsupported_date() -> None:
    tools = valid_word_tools(reopened_suffix="\n\n*汇总生成时间：2025年7月*")
    render_result = tools[-3]["result"]
    assert isinstance(render_result, dict)
    render_data = render_result["data"]
    assert isinstance(render_data, dict)
    render_data["generated_metadata_dates"] = ["2025年7月"]

    result = runner.semantic_checks(
        "word",
        {"status": "completed", "objective": "总结六份 Word", "user_request": "生成汇总"},
        tools,
        "完成并验证 summary.docx",
    )

    assert result["passed"] is False
    assert result["checks"]["reopened_body_nonempty"] is True
    assert result["checks"]["covers_all_six_fixture_topics"] is True
    assert result["checks"]["generated_metadata_dates_supported"] is False


def test_word_gate_normalizes_source_dates_and_rejects_excluded_marker() -> None:
    tools = valid_word_tools(reopened_suffix="\n\n*报告日期：2025年7月*")
    source_result = tools[0]["result"]
    assert isinstance(source_result, dict)
    source_result["data"] = {"date_literals": ["2025年 7月"]}
    render_result = tools[-3]["result"]
    assert isinstance(render_result, dict)
    render_result["data"] = {"path": "summary.docx", "generated_metadata_dates": ["2025年7月"]}

    passed = runner.semantic_checks("word", {"status": "completed"}, tools, "完成 summary.docx")
    assert passed["passed"] is True

    tools[-1]["result"]["stdout"] += runner.EXCLUDED_WORD_MARKER
    failed = runner.semantic_checks("word", {"status": "completed"}, tools, "完成 summary.docx")
    assert failed["passed"] is False
    assert failed["checks"]["excluded_intermediate_marker_absent"] is False


def test_text_gate_requires_topics_and_forbids_managed_writes() -> None:
    final = "项目目标、用户发现、功能范围、安全要求、实施计划、验收指标以及三项主要风险。"
    state = {"status": "completed"}
    read_only = [tool_call(1, 1, "template.read_file", args={"path": "材料.txt"})]

    assert runner.semantic_checks("text", state, read_only, final)["passed"] is True

    with_write = [*read_only, tool_call(1, 2, "file.apply", data={"path": "unexpected.md"})]
    result = runner.semantic_checks("text", state, with_write, final)
    assert result["passed"] is False
    assert result["checks"]["no_managed_write_tools"] is False


def test_session_answer_is_authoritative_and_mismatch_fails_metrics() -> None:
    session_answer = "项目目标、用户发现、功能范围、安全要求、实施计划、验收指标、风险。"
    state = {
        "status": "completed",
        "turn": 1,
        "round": 1,
        "final_answer": session_answer,
        "tool_calls": [tool_call(1, 1, "template.read_file")],
    }
    metrics = runner.build_metrics(
        case_kind="text",
        workspace=Path("/tmp/project"),
        state=state,
        messages=[],
        session_id="session1",
        session_error="",
        screen="assistant\ntruncated\nproject:session1:YOLO > ",
        raw_text="raw",
        screen_answer="truncated",
        timing=timing(),
        process_returncode=0,
        runner_error="",
        timed_out=False,
        final_prompt_seen=True,
    )

    assert metrics["semantic_gate"]["passed"] is True
    assert metrics["screen_answer_matches_session"] is False
    assert metrics["passed"] is False


def test_word_artifact_evidence_copies_inside_workspace_and_redacts_report(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    output = tmp_path / "run"
    workspace.mkdir()
    output.mkdir()
    artifact = workspace / "summary.docx"
    artifact.write_bytes(b"PK\x03\x04bounded-docx-evidence")
    tools = valid_word_tools(
        workspace=workspace,
        reopened_suffix="\nDEEPSEEK_API_KEY=sk-1234567890abcdef",
    )

    evidence = runner._write_word_artifact_evidence(
        workspace=workspace,
        output=output,
        tool_calls=tools,
    )

    assert evidence["passed"] is True
    assert evidence["workspace_relative_source"] == "summary.docx"
    assert (output / "final-artifact.docx").read_bytes() == artifact.read_bytes()
    report = (output / "re-opened-artifact.md").read_text(encoding="utf-8")
    assert "统一数据口径" in report
    assert "sk-1234567890abcdef" not in report
    assert "[REDACTED]" in report


def test_word_artifact_evidence_rejects_workspace_escape_and_oversize(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.docx"
    outside.write_bytes(b"PK-outside")
    escaped_tools = valid_word_tools(workspace=workspace)
    for item in escaped_tools[-3:]:
        request = item["request"]
        result = item["result"]
        assert isinstance(request, dict) and isinstance(result, dict)
        args = request["args"]
        data = result["data"]
        assert isinstance(args, dict) and isinstance(data, dict)
        if "path" in args:
            args["path"] = str(outside)
        if "path" in data:
            data["path"] = str(outside)
    escaped_output = tmp_path / "escaped-run"
    escaped_output.mkdir()

    escaped = runner._write_word_artifact_evidence(
        workspace=workspace,
        output=escaped_output,
        tool_calls=escaped_tools,
    )

    assert escaped["passed"] is False
    assert "outside the workspace" in escaped["error"]
    assert not (escaped_output / "final-artifact.docx").exists()
    assert (escaped_output / "re-opened-artifact.md").is_file()

    artifact = workspace / "summary.docx"
    artifact.write_bytes(b"PK-too-large")
    oversized_output = tmp_path / "oversized-run"
    oversized_output.mkdir()
    monkeypatch.setattr(runner, "WORD_ARTIFACT_MAX_BYTES", 4)
    oversized = runner._write_word_artifact_evidence(
        workspace=workspace,
        output=oversized_output,
        tool_calls=valid_word_tools(workspace=workspace),
    )

    assert oversized["passed"] is False
    assert "exceeds 4 bytes" in oversized["error"]
    assert not (oversized_output / "final-artifact.docx").exists()


def test_main_writes_evidence_when_pty_setup_fails(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    prompt_file = tmp_path / "prompt.txt"
    output = tmp_path / "result"
    prompt_file.write_text("总结材料.txt", encoding="utf-8")
    monkeypatch.setattr(runner.pty, "openpty", lambda: (_ for _ in ()).throw(OSError("PTY unavailable")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--case-kind",
            "text",
        ],
    )

    assert runner.main() == 1
    assert (output / "pty-raw.txt").is_file()
    assert (output / "pty-screen.txt").is_file()
    assert (output / "final-answer.md").is_file()
    metrics = __import__("json").loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["runner_error"] == "OSError: PTY unavailable"
    assert metrics["passed"] is False
