from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import pytest

import agent.tools.base as tool_base
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.state import AgentState
from agent.tools import ToolManager
from agent.tools.base import ToolResult
from agent.tools.registry import ToolCapability


def build_manager(
    root: Path,
    make_config,
    overrides=None,
    *,
    approve=None,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
):
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return (
        config,
        project,
        memory,
        ToolManager(
            config,
            project,
            memory,
            approval_handler=approve,
            auto_approve=auto_approve,
            yolo=yolo,
            super_yolo=super_yolo,
        ),
    )


def test_tool_request_result_and_permission_policy(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    request, result = tools.execute_model_call("shell_run", {"command": "printf ok"})
    assert request.capability == "shell.run"
    assert result.success is True
    assert result.stdout == "ok"
    assert result.request_id == request.request_id
    assert result.duration_ms >= 0

    _, denied_sudo = tools.execute_model_call("shell_run", {"command": "sudo true"})
    _, denied_cwd = tools.execute_model_call("shell_run", {"command": "pwd", "cwd": str(tmp_path.parent)})
    _, denied_timeout = tools.execute_model_call("shell_run", {"command": "true", "timeout": 999})
    assert denied_sudo.success is False
    assert "denied" in denied_sudo.stderr
    assert denied_sudo.data["not_executed"] is True
    assert denied_cwd.success is False
    assert "outside" in denied_cwd.stderr
    assert denied_cwd.data["not_executed"] is True
    assert denied_timeout.success is False
    assert "exceeds" in denied_timeout.stderr
    assert denied_timeout.data["not_executed"] is True

    _, phase_denied = tools.execute_model_call(
        "list_dir",
        {"path": "."},
        runtime_denied_reason="read phase is closed for verification",
    )
    assert phase_denied.success is False
    assert phase_denied.data == {"runtime_denied": True, "not_executed": True}
    assert "verification" in phase_denied.stderr

    _, _, _, guarded = build_manager(root, make_config)
    _, confirmation_denied = guarded.execute_model_call("shell_run", {"command": "printf blocked"})
    assert confirmation_denied.success is False
    assert "requires user confirmation" in confirmation_denied.stderr
    assert confirmation_denied.data["not_executed"] is True


def test_pre_handler_failures_are_marked_without_relabeling_handler_failures(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    _, invalid_json = tools.execute_model_call("list_dir", "{")
    _, unknown = tools.execute_model_call("missing_tool", {})
    _, runtime_denied = tools.execute_model_call(
        "list_dir",
        {"path": "."},
        runtime_denied_reason="read phase is closed",
    )

    entered: list[str] = []
    tools.registry.register(
        ToolCapability(
            "probe",
            "requires_value",
            "probe_requires_value",
            "Probe argument binding.",
            {"value": {"type": "string"}},
            ("value",),
        ),
        lambda value: entered.append(value) or ToolResult(True, value),
    )
    _, invalid_handler_arguments = tools.execute_model_call("probe_requires_value", {})

    def dependency_failure() -> ToolResult:
        entered.append("dependency")
        return ToolResult(False, "", "required dependency is not installed")

    tools.registry.register(
        ToolCapability(
            "probe",
            "dependency",
            "probe_dependency",
            "Probe an executed dependency failure.",
            {},
        ),
        dependency_failure,
    )
    _, executed_failure = tools.execute_model_call("probe_dependency", {})

    def internal_type_error() -> ToolResult:
        entered.append("type-error")
        raise TypeError("dependency API returned an invalid shape")

    tools.registry.register(
        ToolCapability(
            "probe",
            "internal_type_error",
            "probe_internal_type_error",
            "Probe a TypeError after handler entry.",
            {},
        ),
        internal_type_error,
    )
    _, executed_type_error = tools.execute_model_call("probe_internal_type_error", {})

    for result in (invalid_json, unknown, runtime_denied, invalid_handler_arguments):
        assert result.success is False
        assert result.data["not_executed"] is True
    assert runtime_denied.data["runtime_denied"] is True
    assert entered == ["dependency", "type-error"]
    assert executed_failure.success is False
    assert executed_failure.data is None or "not_executed" not in executed_failure.data
    assert executed_type_error.success is False
    assert "invalid shape" in executed_type_error.stderr
    assert executed_type_error.data is None or "not_executed" not in executed_type_error.data


def test_user_approval_denial_is_marked_not_executed(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, approve=lambda *_args: False)

    _, denied = tools.execute_model_call("shell_run", {"command": "printf should-not-run"})

    assert denied.success is False
    assert "denied by user" in denied.stderr
    assert denied.data["not_executed"] is True


def test_shell_pipeline_reports_an_earlier_command_failure(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    _, failed = tools.execute_model_call("shell_run", {"command": "printf failure >&2; false | head -1"})
    _, passed = tools.execute_model_call("shell_run", {"command": "printf success | head -1"})

    assert failed.success is False
    assert failed.data["returncode"] != 0
    assert passed.success is True
    assert passed.stdout == "success"


def test_yolo_and_super_yolo_permission_levels(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, yolo = build_manager(root, make_config, yolo=True)
    capability, _ = yolo.registry.resolve("shell.run")
    request = yolo.registry.request("shell.run", {"command": "sudo true"})
    assert capability is not None
    assert yolo.permission.evaluate(request, capability, super_yolo=False).allowed is False

    _, _, _, super_yolo = build_manager(root, make_config, super_yolo=True)
    capability, _ = super_yolo.registry.resolve("shell.run")
    request = super_yolo.registry.request("shell.run", {"command": "sudo true", "cwd": str(tmp_path.parent)})
    assert capability is not None
    decision = super_yolo.permission.evaluate(request, capability, super_yolo=True)
    assert decision.allowed is True
    assert "SUPER YOLO" in decision.reason


def test_yolo_denies_docker_host_escape_variants(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)
    capability, _ = tools.registry.resolve("docker.run")
    assert capability is not None
    denied = (
        ["run", "--mount", "type=bind,src=/,dst=/host", "alpine"],
        ["run", "--pid=host", "alpine"],
        ["run", "-v=/var/run/docker.sock:/var/run/docker.sock", "alpine"],
        ["run", "--device=/dev/sda", "alpine"],
    )
    for args in denied:
        request = tools.registry.request("docker.run", {"args": args})
        assert tools.permission.evaluate(request, capability).allowed is False


def test_auto_approve_does_not_approve_shell(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, auto_approve=True)
    _, result = tools.execute_model_call("shell_run", {"command": "printf blocked"})
    assert result.success is False
    assert "requires user confirmation" in result.stderr


def test_correction_memory_requires_topic_and_adds_project_tag(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, project, memory, tools = build_manager(root, make_config, yolo=True)

    _, denied = tools.execute_model_call(
        "memory_add",
        {"kind": "Correction", "title": "Port", "content": "Use 8080", "tags": []},
    )
    assert denied.success is False
    _, added = tools.execute_model_call(
        "memory_add",
        {
            "kind": "Correction",
            "title": "Port",
            "content": "Use 8080",
            "tags": ["correction:port"],
        },
    )
    assert added.success is True
    item = memory.get_memory(added.data["id"])
    assert item is not None
    assert project.name in item.tags


def test_capability_config_and_plan_state(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    overrides = {
        "tools": {
            "capabilities": {
                "browser": {"open_url": {"enabled": False}},
                "shell": {"run": {"timeout_seconds": 7}},
            }
        }
    }
    _, project, _, tools = build_manager(root, make_config, overrides)
    names = {item["function"]["name"] for item in tools.schemas()}
    shell_capability = next(item for item in tools.capabilities() if item.name == "shell.run")
    assert "browser_open_url" not in names
    assert shell_capability.timeout_seconds == 7
    assert tools.shell.timeout == 7

    state = AgentState.create(
        session_id="session-1",
        project=project,
        user_request="test plan",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)
    _, plan_result = tools.execute_model_call(
        "agent_update_plan",
        {"steps": [{"id": "inspect", "title": "Inspect files", "status": "in_progress"}]},
    )
    _, step_result = tools.execute_model_call(
        "agent_update_step",
        {"step_id": "inspect", "status": "completed"},
    )
    assert plan_result.success is True
    assert step_result.success is True
    assert state.completed_steps == ["inspect"]
    assert state.current_step is None

    plan_schema = next(item for item in tools.schemas() if item["function"]["name"] == "agent_update_plan")
    assert plan_schema["function"]["parameters"]["properties"]["steps"]["maxItems"] == 8
    _, oversized = tools.execute_model_call(
        "agent_update_plan",
        {"steps": [{"id": f"step-{index}", "title": f"Step {index}"} for index in range(9)]},
    )
    assert oversized.success is False
    assert "at most 8" in oversized.stderr

    tools.plan_manager.replace(
        state,
        [{"id": "required", "title": "Required work", "status": "in_progress"}],
    )
    _, replacement = tools.execute_model_call(
        "agent_update_plan",
        {"steps": [{"id": "done", "title": "Claim done", "status": "completed"}]},
    )
    assert replacement.success is False
    assert "already exists" in replacement.stderr
    assert [step.id for step in state.plan] == ["required"]


def test_read_file_rejects_reversed_line_range(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "example.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    _, result = tools.execute_model_call(
        "read_file",
        {"path": "example.txt", "start_line": 3, "end_line": 1},
    )

    assert result.success is False
    assert "end_line" in result.stderr


def test_read_file_uses_unambiguous_line_number_delimiter(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "example.py").write_text("def sample():\n    return 1\n", encoding="utf-8")
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    _, result = tools.execute_model_call(
        "read_file",
        {"path": "example.py", "start_line": 1, "end_line": 2},
    )

    assert result.success is True
    assert result.stdout.splitlines() == ["     1→def sample():", "     2→    return 1"]
    assert "     2      return 1" not in result.stdout
    schemas = {item["function"]["name"]: item["function"] for item in tools.schemas()}
    assert "prefix through → is display metadata" in schemas["read_file"]["description"]
    assert (
        "excluding every read_file line-number/→ prefix"
        in schemas["file_diff"]["parameters"]["properties"]["old_text"]["description"]
    )


def test_git_tool_in_repository(tmp_path: Path, make_config) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "example.txt").write_text("content\n", encoding="utf-8")
    _, _, _, tools = build_manager(root, make_config)

    _, result = tools.execute_model_call("git_status", {})

    assert result.success is True
    assert "example.txt" in result.stdout


def test_shell_timeout_kills_child_process_group(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)
    marker = root / "orphan.txt"
    command = f"(sleep 2; printf orphan > {marker}) & wait"

    _, result = tools.execute_model_call("shell_run", {"command": command, "timeout": 1})
    time.sleep(2.2)

    assert result.success is False
    assert "timeout" in result.stderr
    assert not marker.exists()


def test_run_command_interrupt_kills_child_before_propagating(tmp_path: Path, monkeypatch) -> None:
    real_popen = subprocess.Popen
    created = []

    class InterruptOnce:
        def __init__(self, *args, **kwargs) -> None:
            self.process = real_popen(*args, **kwargs)
            self.pid = self.process.pid
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            self.stderr = self.process.stderr
            self.returncode = self.process.returncode
            self.interrupted = False
            created.append(self)

        def wait(self, timeout=None):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            value = self.process.wait(timeout=timeout)
            self.returncode = self.process.returncode
            return value

        def poll(self):
            value = self.process.poll()
            self.returncode = self.process.returncode
            return value

    monkeypatch.setattr(tool_base.subprocess, "Popen", InterruptOnce)

    with pytest.raises(KeyboardInterrupt):
        tool_base.run_command(["bash", "-lc", "sleep 30"], cwd=tmp_path, timeout=60)

    assert len(created) == 1
    assert created[0].process.poll() is not None


def test_shell_output_is_bounded(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(
        root,
        make_config,
        {"tools": {"tool_result": {"max_attachment_bytes": 4096}}},
        yolo=True,
    )

    _, result = tools.execute_model_call("shell_run", {"command": "head -c 50000 /dev/zero | tr '\\0' x"})

    assert result.success is True
    assert len(result.stdout.encode()) <= 4096
    assert "source middle omitted" in result.stdout
    assert result.data["source_truncated"] is True
    assert result.data["source_original_bytes"] == 50_000
    assert result.data["source_original_bytes_known"] is True
    assert "attachment" not in result.data
    assert len(ToolResult(True, "x" * 5_000).as_text(limit=512)) == 512


def test_failed_tool_result_bounded_text_is_valid_json_with_failure_evidence() -> None:
    result = ToolResult(
        False,
        "partial stdout " * 200,
        "root cause: injected failure " * 200,
        data={"path": "src/failing.py", "returncode": 17},
        duration_ms=42,
        request_id="failed-call",
    )
    original = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    rendered = result.as_text(limit=512)
    payload = json.loads(rendered)

    assert len(rendered) <= 512
    assert payload["success"] is False
    assert "root cause" in payload["stderr"]
    assert payload["data"] == {"path": "src/failing.py", "returncode": 17}
    assert payload["truncated"] is True
    assert payload["original_chars"] == len(original)
    assert payload["sha256"] == hashlib.sha256(original.encode()).hexdigest()


def test_document_input_size_is_bounded(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    large = root / "large.txt"
    large.write_text("x" * 101, encoding="utf-8")
    _, _, _, tools = build_manager(
        root,
        make_config,
        {"tools": {"document": {"max_input_bytes": 100}}},
        yolo=True,
    )

    _, result = tools.execute_model_call("document_parse", {"path": "large.txt"})

    assert result.success is False
    assert "document exceeds 100 bytes" in result.stderr


def test_document_render_docx_uses_preview_apply_and_parse_verification(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, project, _, tools = build_manager(root, make_config, yolo=True)
    state = AgentState.create(
        session_id="document-session",
        project=project,
        user_request="生成 Word 汇总",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)

    _, rendered = tools.execute_model_call(
        "document_render_docx",
        {
            "path": "结果/汇总.docx",
            "title": "六份材料汇总",
            "markdown": "# 核心结论\n\n- 第一项\n- 第二项\n\n这是经过验证的中文正文。\n\n*汇总生成时间：2025年7月*",
        },
    )

    assert rendered.success is True
    assert rendered.data["requires_apply"] is True
    assert rendered.data["format"] == "docx"
    assert rendered.data["date_literals"] == ["2025年7月"]
    assert rendered.data["generated_metadata_dates"] == ["2025年7月"]
    target = root / "结果" / "汇总.docx"
    assert not target.exists()

    _, applied = tools.execute_model_call("file_apply", {"preview_id": rendered.data["preview_id"]})
    assert applied.success is True
    assert target.read_bytes().startswith(b"PK")

    _, parsed = tools.execute_model_call("document_parse", {"path": "结果/汇总.docx"})
    assert parsed.success is True
    assert "核心结论" in parsed.stdout
    assert "经过验证的中文正文" in parsed.stdout
    assert parsed.data["date_literals"] == ["2025年7月"]


def test_document_render_docx_rejects_path_escape_and_bounded_input(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, project, _, tools = build_manager(
        root,
        make_config,
        {"tools": {"document": {"max_render_chars": 20}}},
        yolo=True,
    )
    state = AgentState.create(
        session_id="document-session",
        project=project,
        user_request="生成 Word 汇总",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)

    _, escaped = tools.execute_model_call(
        "document_render_docx",
        {"path": "../outside.docx", "title": "x", "markdown": "short"},
    )
    _, oversized = tools.execute_model_call(
        "document_render_docx",
        {"path": "inside.docx", "title": "x", "markdown": "中" * 21},
    )

    assert escaped.success is False
    assert "outside the current project" in escaped.stderr
    assert oversized.success is False
    assert "exceeds 20 characters" in oversized.stderr


def test_make_dir_creates_bounded_project_directory(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    _, created = tools.execute_model_call("make_dir", {"path": "结果/子目录"})
    _, escaped = tools.execute_model_call("make_dir", {"path": "../outside"})

    assert created.success is True
    assert created.data["path"] == "结果/子目录"
    assert (root / "结果" / "子目录").is_dir()
    assert escaped.success is False
    assert "outside the current project" in escaped.stderr


def test_document_render_docx_uses_binary_preview_apply_and_parse(tmp_path: Path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr("agent.tools.document.AI_TOOLS_LAUNCHER", tmp_path / "missing-ai-parser-launcher")
    monkeypatch.setattr("agent.tools.document.AI_TOOLS_PARSER", tmp_path / "missing-ai-parser.py")
    _, project, _, tools = build_manager(root, make_config, yolo=True)
    state = AgentState.create(
        session_id="docx-session",
        project=project,
        user_request="create a Word summary",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)

    _, preview = tools.execute_model_call(
        "document_render_docx",
        {
            "path": "汇总.docx",
            "title": "项目汇总",
            "markdown": "# 范围\n\n- 第一项\n- 第二项\n\n结论内容。",
        },
    )
    assert preview.success is True
    assert not (root / "汇总.docx").exists()
    assert preview.data["binary"] is True

    _, applied = tools.execute_model_call("file_apply", {"preview_id": preview.data["preview_id"]})
    assert applied.success is True
    assert (root / "汇总.docx").is_file()

    _, parsed = tools.execute_model_call("document_parse", {"path": "汇总.docx"})
    assert parsed.success is True
    assert parsed.data["parser"] == "pandoc"
    assert "范围" in parsed.stdout
    assert "结论内容" in parsed.stdout

    _, undone = tools.execute_model_call("file_undo", {})
    assert undone.success is True
    assert not (root / "汇总.docx").exists()


def test_run_tests_selects_existing_package_validation_script(tmp_path: Path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.json").write_text(
        json.dumps({"scripts": {"typecheck": "tsc --noEmit", "build": "tsc"}}),
        encoding="utf-8",
    )
    _, _, _, tools = build_manager(root, make_config, yolo=True)
    calls: list[list[str]] = []

    def fake_run_command(args, *, cwd, timeout, **_kwargs):
        calls.append(args)
        return ToolResult(True, "typecheck passed", data={"args": args})

    monkeypatch.setattr("agent.tools.templates.run_command", fake_run_command)

    _, result = tools.execute_model_call("run_tests", {"framework": "auto", "path": "."})

    assert result.success is True
    assert calls == [["npm", "run", "typecheck"]]


def test_run_tests_never_invokes_missing_npm_test_script(tmp_path: Path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.json").write_text(json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8")
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a missing package script must not execute")

    monkeypatch.setattr("agent.tools.templates.run_command", forbidden)

    _, result = tools.execute_model_call("run_tests", {"framework": "auto", "path": "."})

    assert result.success is False
    assert "available: none" in result.stderr


def test_run_tests_prefers_explicit_project_markers_over_generic_tests_directory(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)
    cases = (
        ("rust", "Cargo.toml", "cargo"),
        ("go", "go.mod", "go"),
        ("gradle", "gradlew", "gradle"),
        ("maven", "pom.xml", "maven"),
    )

    for directory, marker, expected in cases:
        project = root / directory
        project.mkdir()
        (project / "tests").mkdir()
        (project / marker).touch()
        assert tools.templates._detect_framework(project) == expected

    generic = root / "generic"
    generic.mkdir()
    (generic / "tests").mkdir()
    assert tools.templates._detect_framework(generic) == "pytest"


def test_implement_step_can_skip_only_for_conditional_mutation(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, project, _, tools = build_manager(root, make_config, yolo=True)
    state = AgentState.create(
        session_id="conditional-plan",
        project=project,
        user_request="only fix a proven bug",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)
    tools.plan_manager.replace(
        state,
        [
            {"id": "scope", "title": "scope", "status": "completed"},
            {
                "id": "inspect-chunks",
                "title": "inspect",
                "status": "completed",
                "dependencies": ["scope"],
            },
            {"id": "implement", "title": "implement", "dependencies": ["inspect-chunks"]},
            {"id": "verify", "title": "verify", "dependencies": ["implement"]},
        ],
    )

    _, denied = tools.execute_model_call("agent_update_step", {"step_id": "implement", "status": "skipped"})
    state.task_route = {"reasons": ["mutation-request", "conditional-mutation"]}
    _, allowed = tools.execute_model_call("agent_update_step", {"step_id": "implement", "status": "skipped"})
    _, scope_denied = tools.execute_model_call("agent_update_step", {"step_id": "scope", "status": "skipped"})
    _, verify_denied = tools.execute_model_call("agent_update_step", {"step_id": "verify", "status": "skipped"})

    assert denied.success is False
    assert allowed.success is True
    assert scope_denied.success is False
    assert verify_denied.success is False
    assert "only the implement step" in scope_denied.stderr
    assert "only the implement step" in verify_denied.stderr
    plan = {step.id: step for step in state.plan}
    assert plan["scope"].status == "completed"
    assert plan["implement"].status == "skipped"
    assert plan["verify"].status == "pending"
    assert state.plan_step_satisfied(plan["implement"]) is True
    assert state.plan_step_satisfied(plan["verify"]) is False
    assert [step.id for step in tools.plan_manager.ready_steps(state)] == ["verify"]
    assert state.current_step == "verify"


def test_update_plan_cannot_inject_an_invalid_skipped_step(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, project, _, tools = build_manager(root, make_config, yolo=True)
    state = AgentState.create(
        session_id="invalid-skipped-plan",
        project=project,
        user_request="inspect and verify",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {"reasons": ["conditional-mutation"]}
    tools.bind_state(state)

    _, result = tools.execute_model_call(
        "agent_update_plan",
        {"steps": [{"id": "verify", "title": "verify", "status": "skipped"}]},
    )

    assert result.success is False
    assert "only the implement step" in result.stderr
    assert state.plan == []
