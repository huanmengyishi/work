from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent.parallel import ParallelWorktreeRunner
from agent.project import ProjectManager
from agent.task_queue import TaskQueueManager


def test_task_queue_resume_skips_completed_tasks(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    queues = TaskQueueManager(project)
    record = queues.create(["one", "two", "three"])
    calls: list[str] = []

    def first_runner(task, queue_record):
        calls.append(task.prompt)
        if task.prompt == "two":
            return "failed", "session-two", "failed"
        return f"done {task.prompt}", f"session-{task.prompt}", "completed"

    queues.run(record, first_runner, stop_on_failure=True)
    assert record.status == "paused"
    assert [task.status for task in record.tasks] == ["completed", "failed", "pending"]

    def resume_runner(task, queue_record):
        calls.append(task.prompt)
        return f"done {task.prompt}", f"session-{task.prompt}", "completed"

    queues.run(queues.load(record.id), resume_runner)
    restored = queues.load(record.id)
    assert restored.status == "completed"
    assert calls == ["one", "two", "two", "three"]


def test_parallel_threshold_dirty_guard_and_patch_merge(tmp_path: Path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    runner = ParallelWorktreeRunner(project, config.data_dir)
    with pytest.raises(ValueError, match="at least 8"):
        runner.run(["one"])

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_agent = fake_bin / "agent"
    fake_agent.write_text(
        '#!/bin/sh\nn=$(basename "$PWD" | sed \'s/task-//\')\nprintf \'task %s\\n\' "$n" > "result-$n.txt"\n',
        encoding="utf-8",
    )
    fake_agent.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    run_id, results = runner.run([f"task {index}" for index in range(8)], max_workers=4)
    assert run_id
    assert all(result.applied for result in results)
    assert sorted(path.name for path in root.glob("result-*.txt")) == [f"result-{index}.txt" for index in range(1, 9)]

    (root / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git"):
        runner.run([f"task {index}" for index in range(8)])
