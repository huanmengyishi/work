from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.context import ContextBuildRequest, ContextBuilder, ContextPackage
from agent.deepseek import ChatResponse
from agent.events import EventBus
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.prompt import PromptBuilder
from agent.runtime import AgentRuntime
from agent.state import AgentState, PlanStep
from agent.tools import ToolManager


class RecordingContextBuilder(ContextBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.packages: list[ContextPackage] = []

    def build_package(self, request: ContextBuildRequest) -> ContextPackage:
        package = super().build_package(request)
        self.packages.append(package)
        return package


class RecordingPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        self.packages: list[ContextPackage] = []

    def build_initial(self, package: ContextPackage) -> list[dict[str, object]]:
        self.packages.append(package)
        return super().build_initial(package)

    def build_resume(self, package: ContextPackage) -> list[dict[str, object]]:
        self.packages.append(package)
        return super().build_resume(package)


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def chat(
        self,
        *,
        messages,
        tools=None,
        tool_choice="auto",
        max_tokens=None,
        thinking=None,
        reasoning_effort=None,
        model=None,
    ) -> ChatResponse:
        if not self.responses:
            raise AssertionError("fake response queue exhausted")
        return ChatResponse(message=self.responses.pop(0), raw={})


class RecordingClient(FakeClient):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(responses)
        self.requests: list[list[dict]] = []
        self.options: list[dict] = []
        self.models: list[str | None] = []

    def chat(
        self,
        *,
        messages,
        tools=None,
        tool_choice="auto",
        max_tokens=None,
        thinking=None,
        reasoning_effort=None,
        model=None,
    ) -> ChatResponse:
        self.requests.append(list(messages))
        self.options.append({"thinking": thinking, "reasoning_effort": reasoning_effort})
        self.models.append(model)
        return super().chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            model=model,
        )


def test_runtime_builds_context_package_before_initial_and_resume_prompt(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    context_builder = RecordingContextBuilder(config)
    prompt_builder = RecordingPromptBuilder()
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient(
            [
                {"role": "assistant", "content": "initial complete"},
                {"role": "assistant", "content": "resume complete"},
            ]
        ),
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    assert runtime.run("explain this project") == "initial complete"
    assert runtime.resume("continue", runtime.last_session_id) == "resume complete"
    assert runtime.sessions.list_sessions(limit=1)[0].user_request == "explain this project"

    assert [package.phase for package in context_builder.packages] == ["initial", "resume"]
    assert prompt_builder.packages == context_builder.packages
    assert all(isinstance(package, ContextPackage) for package in prompt_builder.packages)


def tool_message(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_runtime_checkpoint_resume_events_and_memory_pipeline(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    tools = ToolManager(config, project, memory, yolo=True)
    client = FakeClient(
        [
            tool_message(
                "plan-1",
                "agent_update_plan",
                {"steps": [{"id": "inspect", "title": "Inspect", "status": "in_progress"}]},
            ),
            tool_message("shell-1", "shell_run", {"command": "printf first-turn"}),
            {"role": "assistant", "content": "first complete"},
            tool_message("shell-2", "shell_run", {"command": "printf second-turn"}),
            {"role": "assistant", "content": "second complete"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("inspect the project") == "first complete"
    session_id = runtime.last_session_id
    assert session_id
    assert runtime.resume("continue verification", session_id) == "second complete"

    session_path = project.agent_dir / "sessions" / f"{session_id}.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    state = payload["state"]
    assert state["status"] == "completed"
    assert state["schema_version"] == 6
    assert state["task_route"]["mode"] == "standard"
    assert state["model_route"]["provider"] == "deepseek"
    assert state["context_manifest"]["used_chars"] <= state["context_manifest"]["max_chars"]
    assert state["turn"] == 2
    assert [item["turn"] for item in state["tool_calls"]] == [1, 1, 2]
    assert state["tool_calls"][-1]["result"]["stdout"] == "second-turn"
    assert (project.agent_dir / "sessions" / f"{session_id}.md").exists()
    assert (project.agent_dir / "index.json").exists()

    memories = memory.recent(project.id, limit=20)
    assert [item.kind for item in memories].count("Summary") == 2
    assert [item.kind for item in memories].count("Lesson") == 2
    with sqlite3.connect(memory.db_path) as con:
        assert con.execute("select count(*) from pipeline_runs").fetchone()[0] == 2

    before = len(memory.recent(project.id, limit=20))
    runtime._publish_terminal("task.finished", runtime.sessions.load(session_id).state, final="duplicate")
    after = len(memory.recent(project.id, limit=20))
    assert after == before


def test_runtime_injects_recovery_memory_after_tool_failure(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    memory.add_memory(
        kind="Correction",
        title="Missing command",
        content="Install missing-command before retrying.",
        tags=["correction:dependency", project.name],
        project_id=project.id,
    )
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            tool_message("shell-1", "shell_run", {"command": "missing-command --version"}),
            {"role": "assistant", "content": "diagnosed"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)
    assert runtime.run("run the missing command") == "diagnosed"
    assert any("Failure Recovery Memory" in str(message.get("content")) for message in client.requests[1])


def test_runtime_adapts_deep_task_and_reports_reasoning_progress(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            tool_message("step-1", "agent_update_step", {"step_id": "scope", "status": "completed"}),
            tool_message("inspect-file", "read_file", {"path": "main.py", "start_line": 1, "end_line": 20}),
            tool_message(
                "step-2",
                "agent_update_step",
                {"step_id": "inspect-chunks", "status": "completed"},
            ),
            tool_message("step-3", "agent_update_step", {"step_id": "implement", "status": "completed"}),
            tool_message("step-4", "agent_update_step", {"step_id": "verify", "status": "completed"}),
            {
                "role": "assistant",
                "content": "complete",
                "reasoning_content": "inspect bounded chunks first",
            },
        ]
    )
    progress: list[dict] = []
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
        progress_handler=progress.append,
    )

    assert runtime.run("全面审计整个代码库的所有安全问题并完成大规模重构") == "complete"
    session = runtime.sessions.load(runtime.last_session_id)

    assert session.state.task_strategy["mode"] == "deep"
    assert session.state.task_route["task_type"] == "refactor"
    assert session.state.task_route["risk"] == "high"
    assert session.state.model_route["provider"] == "deepseek"
    assert session.state.model_route["tier"] == "deep"
    assert session.state.task_strategy["max_tool_rounds"] == 24
    assert [step.id for step in session.state.plan] == ["scope", "inspect-chunks", "implement", "verify"]
    assert all(item == {"thinking": True, "reasoning_effort": "max"} for item in client.options)
    assert set(client.models) == {"deepseek-v4-pro"}
    assert any(item["event"] == "thinking.content" for item in progress)


def test_runtime_rejects_progress_note_then_accepts_completed_answer(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            {"role": "assistant", "content": "源码在 ~/AI-Agent，需要用 shell"},
            tool_message("list-1", "list_dir", {"path": ".", "depth": 1}),
            {"role": "assistant", "content": "已检查当前项目，并给出基于文件列表的建议。"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("查看一下当前项目，给出后续修改建议") == "已检查当前项目，并给出基于文件列表的建议。"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert len(state.tool_calls) == 1


def test_document_completion_requires_applied_and_reparsed_docx(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    state = AgentState.create(
        session_id="document-gate",
        project=project,
        user_request="总结所有文档，新建 Word 汇总",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {
        "schema_version": 2,
        "reasons": ["artifact-required", "word-artifact-required"],
        "artifact_hints": ["汇总.docx"],
    }
    state.plan = []

    state.tool_calls = [
        {
            "round": 1,
            "request": {"tool": "document", "action": "render_docx"},
            "result": {"success": True, "data": {"path": "汇总.docx", "preview_id": "word-preview"}},
        }
    ]
    assert "managed-write evidence" in AgentRuntime._completion_issue(state, "已生成。")

    state.tool_calls.append(
        {
            "round": 2,
            "request": {"tool": "file", "action": "apply", "args": {"preview_id": "word-preview"}},
            "result": {
                "success": True,
                "data": {
                    "path": "无关.docx",
                    "preview_id": "word-preview",
                    "snapshot_id": "word-snapshot",
                    "before_exists": False,
                    "after_exists": True,
                },
            },
        }
    )
    assert "matching: 汇总.docx" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls[-1]["result"]["data"]["path"] = "汇总.docx"
    state.tool_calls[-1]["request"]["args"]["preview_id"] = "other-preview"
    state.tool_calls[-1]["result"]["data"]["preview_id"] = "other-preview"
    assert "preview_id" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls[-1]["request"]["args"]["preview_id"] = "word-preview"
    state.tool_calls[-1]["result"]["data"]["preview_id"] = "word-preview"
    assert "re-opened" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls.insert(
        1,
        {
            "round": 1,
            "request": {"tool": "document", "action": "render_docx"},
            "result": {"success": True, "data": {"path": "汇总.docx", "preview_id": "newer-preview"}},
        },
    )
    assert "latest generated document preview" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls.pop(1)

    state.tool_calls.append(
        {
            "round": 3,
            "request": {"tool": "document", "action": "parse", "args": {"path": "其他.docx"}},
            "result": {"success": True},
        }
    )
    assert "re-opened" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls[-1]["request"]["args"]["path"] = str(root.parent / "other" / root.name / "汇总.docx")
    assert "re-opened" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls[-1]["request"]["args"]["path"] = str(root / "汇总.docx")
    assert AgentRuntime._completion_issue(state, "已生成并重新打开验证。") == ""
    state.tool_calls.append(
        {
            "round": 4,
            "request": {"tool": "file", "action": "apply", "args": {"preview_id": "delete-word"}},
            "result": {
                "success": True,
                "data": {
                    "path": "汇总.docx",
                    "preview_id": "delete-word",
                    "snapshot_id": "delete-word-snapshot",
                    "before_exists": True,
                    "after_exists": False,
                },
            },
        }
    )
    assert "no active" in AgentRuntime._completion_issue(state, "已生成。")
    state.tool_calls.pop()
    state.tool_calls[0]["result"]["data"]["generated_metadata_dates"] = ["2025年7月"]
    assert "unsupported generation-date metadata" in AgentRuntime._completion_issue(state, "已生成。")

    original_request = state.user_request
    state.user_request = original_request + "，材料日期为2025年 7月"
    assert AgentRuntime._completion_issue(state, "已生成并重新打开验证。") == ""
    state.user_request = original_request

    original_objective = state.objective
    state.objective = original_objective + "，材料日期为2025年 7月"
    assert AgentRuntime._completion_issue(state, "已生成并重新打开验证。") == ""
    state.objective = original_objective

    state.tool_calls.insert(
        0,
        {
            "round": 0,
            "request": {"tool": "document", "action": "parse", "args": {"path": "源材料.docx"}},
            "result": {
                "success": True,
                "stdout": "项目计划日期为2025年 7月",
                "data": {"date_literals": ["2025年 7月"]},
            },
        },
    )
    assert AgentRuntime._completion_issue(state, "已生成并重新打开验证。") == ""

    source = state.tool_calls[0]
    source["request"] = {"tool": "template", "action": "read_file", "args": {"path": "源材料.txt"}}
    source["result"] = {"success": True, "stdout": "项目计划日期为2025年 7月", "data": {}}
    assert AgentRuntime._completion_issue(state, "已生成并重新打开验证。") == ""

    source["request"] = {"tool": "ocr", "action": "parse", "args": {"path": "扫描件.png"}}
    source["result"] = {"success": True, "stdout": "", "data": {"date_literals": ["2025年 7月"]}}
    assert AgentRuntime._completion_issue(state, "已生成并重新打开验证。") == ""

    state.tool_calls.append(state.tool_calls.pop(0))
    assert "unsupported generation-date metadata" in AgentRuntime._completion_issue(state, "已生成。")


def test_artifact_completion_requires_all_named_and_extension_hints(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="artifact-hints",
        project=project,
        user_request="Create summary.md and export a PDF report",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {
        "reasons": ["artifact-required"],
        "artifact_hints": ["summary.md", ".pdf"],
    }
    state.tool_calls = [
        {
            "round": 1,
            "request": {"tool": "file", "action": "apply"},
            "result": {"success": True, "data": {"path": "src/summary.py"}},
        },
        {
            "round": 2,
            "request": {"tool": "file", "action": "apply"},
            "result": {"success": True, "data": {"path": "reports/other.md"}},
        },
    ]

    issue = AgentRuntime._execution_evidence_issue(state)
    assert "summary.md" in issue

    state.tool_calls.append(
        {
            "round": 3,
            "request": {"tool": "file", "action": "apply"},
            "result": {"success": True, "data": {"path": "reports/summary.md"}},
        }
    )
    assert ".pdf" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "round": 4,
            "request": {"tool": "file", "action": "apply"},
            "result": {"success": True, "data": {"path": "reports/final.PDF"}},
        }
    )
    assert AgentRuntime._execution_evidence_issue(state) == ""

    state.task_route["artifact_hints"] = []
    state.tool_calls = state.tool_calls[:1]
    assert AgentRuntime._execution_evidence_issue(state) == ""


def test_artifact_completion_rejects_delete_and_undone_write_but_accepts_directory(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="artifact-final-state",
        project=project,
        user_request="Create summary.md",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {
        "schema_version": 2,
        "reasons": ["artifact-required"],
        "artifact_hints": ["summary.md"],
    }
    state.tool_calls = [
        {
            "round": 1,
            "request": {"tool": "file", "action": "apply", "args": {"preview_id": "delete-preview"}},
            "result": {
                "success": True,
                "data": {
                    "path": "summary.md",
                    "preview_id": "delete-preview",
                    "snapshot_id": "delete-snapshot",
                    "before_exists": True,
                    "after_exists": False,
                },
            },
        }
    ]
    assert "no active" in AgentRuntime._execution_evidence_issue(state)

    state.tool_calls[0]["result"]["data"].update(
        {"preview_id": "create-preview", "snapshot_id": "create-snapshot", "after_exists": True}
    )
    assert AgentRuntime._execution_evidence_issue(state) == ""
    state.tool_calls.append(
        {
            "round": 2,
            "request": {"tool": "file", "action": "apply", "args": {"preview_id": "later-delete"}},
            "result": {
                "success": True,
                "data": {
                    "path": "summary.md",
                    "preview_id": "later-delete",
                    "snapshot_id": "delete-after-create",
                    "before_exists": True,
                    "after_exists": False,
                },
            },
        }
    )
    assert "no active" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "round": 3,
            "request": {"tool": "file", "action": "undo", "args": {"snapshot_id": "delete-after-create"}},
            "result": {
                "success": True,
                "data": {"path": "summary.md", "snapshot_id": "delete-after-create", "restored_exists": True},
            },
        }
    )
    assert AgentRuntime._execution_evidence_issue(state) == ""
    state.tool_calls.append(
        {
            "round": 4,
            "request": {"tool": "file", "action": "undo", "args": {"snapshot_id": "create-snapshot"}},
            "result": {
                "success": True,
                "data": {"path": "summary.md", "snapshot_id": "create-snapshot", "restored_exists": False},
            },
        }
    )
    assert "no active" in AgentRuntime._execution_evidence_issue(state)

    state.task_route = {
        "schema_version": 2,
        "reasons": ["artifact-required", "directory-artifact-required"],
        "artifact_hints": ["reports"],
        "directory_hints": ["reports"],
    }
    assert "make_dir" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "round": 3,
            "request": {"tool": "template", "action": "make_dir", "args": {"path": "unrelated"}},
            "result": {"success": True, "data": {"path": "unrelated"}},
        }
    )
    assert "matching: reports" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "round": 4,
            "request": {"tool": "template", "action": "make_dir", "args": {"path": "reports"}},
            "result": {"success": True, "data": {"path": "reports"}},
        }
    )
    assert AgentRuntime._execution_evidence_issue(state) == ""


def test_soft_target_stays_open_until_named_artifact_is_applied(
    tmp_path: Path,
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 1,
                "max_tool_rounds_hard_limit": 3,
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    monkeypatch.setattr("agent.tools.file_edit.uuid4", lambda: SimpleNamespace(hex="fixed-artifact-id"))
    client = RecordingClient(
        [
            tool_message(
                "preview",
                "file_diff",
                {"path": "summary.md", "content": "# Verified summary\n"},
            ),
            tool_message("apply", "file_apply", {"preview_id": "fixed-artifact-id"}),
            {"role": "assistant", "content": "Created the requested summary.md through the managed workflow."},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)
    prompt = "Create the report file summary.md"
    forced_route = replace(runtime.task_router.route(prompt), mode="standard", max_tool_rounds=1)
    monkeypatch.setattr(runtime.task_router, "route", lambda *_args, **_kwargs: forced_route)

    answer = runtime.run(
        prompt,
        initial_plan=[{"id": "deliver", "title": "Deliver the report", "status": "completed"}],
    )

    assert answer == "Created the requested summary.md through the managed workflow."
    assert (root / "summary.md").read_text(encoding="utf-8") == "# Verified summary\n"
    assert len(client.requests) == 3
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert state.main_loop_model_request_count == 2
    assert state.final_synthesis_model_request_count == 1
    assert [item["request"]["action"] for item in state.tool_calls] == ["diff", "apply"]


def test_missing_artifact_uses_hard_budget_before_tool_free_failure(
    tmp_path: Path,
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 1,
                "max_tool_rounds_hard_limit": 2,
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            tool_message("inspect-1", "list_dir", {"path": ".", "depth": 1}),
            tool_message("inspect-2", "list_dir", {"path": ".", "depth": 1}),
            {"role": "assistant", "content": "The requested artifact was not created."},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    prompt = "Create the report file summary.md"
    forced_route = replace(runtime.task_router.route(prompt), mode="standard", max_tool_rounds=1)
    monkeypatch.setattr(runtime.task_router, "route", lambda *_args, **_kwargs: forced_route)

    answer = runtime.run(
        prompt,
        initial_plan=[{"id": "deliver", "title": "Deliver the report", "status": "completed"}],
    )

    assert "hard tool-turn limit was reached" in answer
    assert "managed-write evidence" in answer
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.error.startswith("hard_limit reached:")
    assert state.main_loop_model_request_count == 2
    assert state.final_synthesis_model_request_count == 1


def test_completion_gate_requires_non_plan_evidence_and_explicit_single_validation(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="evidence-gate",
        project=project,
        user_request="只运行一次静态检查；没有真实缺陷就不要修改",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {
        "task_type": "bug_fix",
        "require_plan": True,
        "reasons": ["mutation-request", "conditional-mutation", "single-validation"],
    }
    state.plan = [
        PlanStep(id="scope", title="scope", status="completed"),
        PlanStep(id="inspect-chunks", title="inspect", status="completed", dependencies=["scope"]),
        PlanStep(id="implement", title="implement", status="skipped", dependencies=["inspect-chunks"]),
        PlanStep(id="verify", title="verify", status="completed", dependencies=["implement"]),
    ]

    assert "no executed non-plan tool evidence" in AgentRuntime._completion_issue(state, "检查完成，无需修改。")

    state.tool_calls.append(
        {
            "request": {"tool": "template", "action": "read_file", "args": {"path": "src/app.ts"}},
            "result": {"success": True, "data": {}},
        }
    )
    assert "no executed validation attempt" in AgentRuntime._completion_issue(state, "检查完成，无需修改。")

    state.tool_calls.append(
        {
            "request": {"tool": "template", "action": "run_tests", "args": {"framework": "npm:typecheck"}},
            "result": {"success": False, "stderr": "existing baseline failure", "data": {"returncode": 2}},
        }
    )
    assert AgentRuntime._completion_issue(state, "检查已执行但存在基线错误；没有独立缺陷，因此未修改。") == ""


def test_single_validation_evidence_is_required_without_a_task_graph(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="single-validation-no-plan",
        project=project,
        user_request="Run validation exactly once and report the result",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {"require_plan": False, "reasons": ["single-validation"]}

    assert "no executed validation attempt" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "request": {"tool": "template", "action": "read_file", "args": {"path": "package.json"}},
            "result": {"success": True, "data": {}},
        }
    )
    assert "no executed validation attempt" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "request": {"tool": "template", "action": "run_tests", "args": {"framework": "npm:test"}},
            "result": {"success": False, "data": {"runtime_denied": True}},
        }
    )
    assert "no executed validation attempt" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "request": {"tool": "template", "action": "run_tests", "args": {"framework": "npm:test"}},
            "result": {"success": False, "stderr": "permission denied", "data": {"not_executed": True}},
        }
    )
    assert AgentRuntime._single_validation_used(state) is False
    assert "no executed validation attempt" in AgentRuntime._execution_evidence_issue(state)
    state.tool_calls.append(
        {
            "request": {"tool": "template", "action": "run_tests", "args": {"framework": "npm:test"}},
            "result": {"success": False, "stderr": "baseline failure", "data": {"returncode": 2}},
        }
    )
    assert AgentRuntime._execution_evidence_issue(state) == ""


def test_completion_gate_reports_plan_and_artifact_gaps_together(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="aggregate-completion-gaps",
        project=project,
        user_request="Create the report file summary.md",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.task_route = {
        "require_plan": True,
        "reasons": ["artifact-required"],
        "artifact_hints": ["summary.md"],
    }

    issue = AgentRuntime._completion_issue(state, "Done.")
    assert "requires a Task Graph" in issue
    assert "managed-write evidence" in issue


def test_runtime_uses_separate_final_synthesis_after_last_tool_round(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            tool_message("list-1", "list_dir", {"path": ".", "depth": 1}),
            {"role": "assistant", "content": "最后一轮工具结果已经综合完成。"},
        ]
    )
    events = EventBus()
    model_events = []
    events.subscribe("model.requested", model_events.append, name="test.model-requested")
    events.subscribe("model.responded", model_events.append, name="test.model-responded")
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
        events=events,
    )

    assert runtime.run("检查项目") == "最后一轮工具结果已经综合完成。"
    assert len(client.requests) == 2
    assert any("Tool execution budget is closed" in str(item.get("content")) for item in client.requests[-1])
    assert [item.name for item in model_events] == [
        "model.requested",
        "model.responded",
        "model.requested",
        "model.responded",
    ]
    assert model_events[-2].payload["phase"] == "final_synthesis"
    assert model_events[-1].payload["phase"] == "final_synthesis"


def test_resume_package_preserves_original_objective_and_tool_evidence(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            tool_message("list-1", "list_dir", {"path": ".", "depth": 1}),
            {"role": "assistant", "content": "first turn complete"},
            {"role": "assistant", "content": "continued"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    objective = "总结当前目录材料并给出建议"
    assert runtime.run(objective) == "first turn complete"
    session_id = runtime.last_session_id
    assert runtime.resume("继续", session_id) == "continued"

    resume_messages = client.requests[-1]
    rendered = "\n".join(str(item.get("content") or "") for item in resume_messages)
    assert objective in rendered
    assert "template.list_dir success" in rendered


def test_short_resume_keeps_deep_strategy_and_plan(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            tool_message("scope-complete", "agent_update_step", {"step_id": "scope", "status": "completed"}),
            tool_message("inspect-project", "list_dir", {"path": ".", "depth": 1}),
            tool_message(
                "inspect-complete",
                "agent_update_step",
                {"step_id": "inspect-chunks", "status": "completed"},
            ),
            tool_message(
                "implement-complete",
                "agent_update_step",
                {"step_id": "implement", "status": "completed"},
            ),
            tool_message("verify-complete", "agent_update_step", {"step_id": "verify", "status": "completed"}),
            {"role": "assistant", "content": "checkpoint"},
            {"role": "assistant", "content": "continued"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("全面审计整个代码库并深度重构所有安全问题") == "checkpoint"
    session_id = runtime.last_session_id
    original_state = runtime.sessions.load(session_id).state
    original_plan = [step.id for step in original_state.plan]
    original_model = original_state.model_route["model"]
    assert runtime.resume("继续", session_id) == "continued"
    resumed = runtime.sessions.load(session_id).state

    assert resumed.task_strategy["mode"] == "deep"
    assert resumed.task_route["mode"] == "deep"
    assert resumed.model_route["tier"] == "deep"
    assert resumed.model_route["model"] == original_model
    assert resumed.context_manifest["phase"] == "resume"
    assert [step.id for step in resumed.plan] == original_plan
    assert client.options[-1] == {"thinking": True, "reasoning_effort": "max"}
    assert client.models == [original_model] * 7


def test_concurrent_resume_rejects_second_session_turn(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    started = threading.Event()
    release = threading.Event()

    class BlockingClient(FakeClient):
        def chat(self, **kwargs) -> ChatResponse:
            started.set()
            assert release.wait(timeout=5)
            return super().chat(**kwargs)

    initial_runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient([{"role": "assistant", "content": "checkpoint"}]),
    )
    assert initial_runtime.run("create resumable session") == "checkpoint"
    session_id = initial_runtime.last_session_id
    first = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=BlockingClient([{"role": "assistant", "content": "first"}]),
    )
    second = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient([{"role": "assistant", "content": "second"}]),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(first.resume, "first continuation", session_id)
        assert started.wait(timeout=5)
        with pytest.raises(RuntimeError, match="already being resumed"):
            second.resume("second continuation", session_id)
        release.set()
        assert future.result(timeout=5) == "first"

    state = first.sessions.load(session_id).state
    assert state.turn == 2
    assert state.final_answer == "first"


def test_runtime_respects_small_context_hard_limit_and_input_limit(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "context": {
                "package_limits": {"simple": 900},
                "max_package_chars_hard_limit": 500,
            },
            "runtime": {"max_user_request_chars": 20},
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([{"role": "assistant", "content": "bounded"}])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("什么是 Python？") == "bounded"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.context_manifest["max_chars"] == 500
    assert state.context_manifest["used_chars"] <= 500
    assert len(client.requests[0][1]["content"]) + len(client.requests[0][2]["content"]) <= 500
    with pytest.raises(ValueError, match="save large text/code"):
        runtime.run("x" * 21)


def test_resume_repeated_failure_escalates_model_without_losing_task_type(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            {"role": "assistant", "content": "checkpoint"},
            {"role": "assistant", "content": "recovered"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    prompt = "修复这个错误并运行测试 " + "详细范围 " * 150
    assert runtime.run(prompt) == "checkpoint"
    session_id = runtime.last_session_id
    record = runtime.sessions.load(session_id)
    original_type = record.state.task_route["task_type"]
    assert record.state.task_route["score"] == 2
    record.state.tool_calls.extend(
        [
            {"result": {"success": False}},
            {"result": {"success": False}},
        ]
    )
    runtime.sessions.checkpoint(record.state, record.messages)

    assert runtime.resume("继续", session_id) == "recovered"
    resumed = runtime.sessions.load(session_id).state
    assert resumed.task_route["task_type"] == original_type
    assert resumed.task_route["failure_count"] == 2
    assert resumed.model_route["tier"] == "deep"
    assert client.models[-1] == resumed.model_route["model"]


def test_runtime_falls_back_from_malformed_context_limits(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "context": {
                "package_limits": {"simple": None},
                "max_package_chars_hard_limit": "invalid",
                "max_recovery_context_chars": None,
            },
            "runtime": {"max_user_request_chars": "invalid"},
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient([{"role": "assistant", "content": "ok"}]),
    )

    assert runtime.run("什么是 Python？") == "ok"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.context_manifest["max_chars"] == 12_000
    assert (
        runtime._bounded_config_int("context.max_recovery_context_chars", 6_000, minimum=0, maximum=1_000_000) == 6_000
    )


def test_resume_keeps_large_scope_but_upgrades_architecture_model(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            tool_message("scope-complete", "agent_update_step", {"step_id": "scope", "status": "completed"}),
            tool_message("inspect-project", "list_dir", {"path": ".", "depth": 1}),
            tool_message(
                "inspect-complete",
                "agent_update_step",
                {"step_id": "inspect-chunks", "status": "completed"},
            ),
            tool_message(
                "synthesize-complete",
                "agent_update_step",
                {"step_id": "synthesize", "status": "completed"},
            ),
            tool_message("verify-complete", "agent_update_step", {"step_id": "verify", "status": "completed"}),
            {"role": "assistant", "content": "large checkpoint"},
            {"role": "assistant", "content": "architecture complete"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("分析整个代码库的所有文件并总结") == "large checkpoint"
    session_id = runtime.last_session_id
    first = runtime.sessions.load(session_id).state
    assert first.task_route["mode"] == "large"
    assert first.model_route["tier"] == "standard"

    assert runtime.resume("请解释并设计系统架构", session_id) == "architecture complete"
    resumed = runtime.sessions.load(session_id).state
    assert resumed.task_route["mode"] == "large"
    assert resumed.model_route["tier"] == "deep"
    assert client.options[-1]["reasoning_effort"] == "max"
