from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from .project import Project
from .paths import storage_key
from .timeutil import utc_now_iso


@dataclass
class ParallelTaskResult:
    index: int
    prompt: str
    status: str
    worktree: str
    patch_path: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    applied: bool = False


class ParallelWorktreeRunner:
    def __init__(self, project: Project, data_dir: Path) -> None:
        self.project = project
        self.base_dir = data_dir / "worktrees" / storage_key(project.id)
        self.report_dir = project.agent_dir / "parallel"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        prompts: list[str],
        *,
        min_tasks: int = 8,
        max_workers: int = 4,
        agent_flags: list[str] | None = None,
    ) -> tuple[str, list[ParallelTaskResult]]:
        tasks = [prompt.strip() for prompt in prompts if prompt.strip()]
        if len(tasks) < min_tasks:
            raise ValueError(f"parallel execution requires at least {min_tasks} independent tasks")
        self._require_clean_git()
        agent_command = shutil.which("agent")
        if not agent_command:
            raise RuntimeError("agent executable is not available on PATH")
        run_id = f"{utc_now_iso().replace(':', '').replace('-', '')}-{uuid4().hex[:8]}"
        run_dir = self.base_dir / run_id
        patch_dir = self.report_dir / run_id
        run_dir.mkdir(parents=True)
        patch_dir.mkdir(parents=True)
        base_commit = self._git("rev-parse", "HEAD").strip()
        specs: list[tuple[int, str, Path, str]] = []
        try:
            for index, prompt in enumerate(tasks, start=1):
                branch = f"agent-parallel/{run_id}/{index}"
                worktree = run_dir / f"task-{index}"
                self._git("worktree", "add", "-q", "-b", branch, str(worktree), "HEAD")
                specs.append((index, prompt, worktree, branch))
        except Exception:
            self._cleanup_worktrees(specs, run_dir)
            raise

        results: list[ParallelTaskResult] = []
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(specs)))) as executor:
                futures = {
                    executor.submit(
                        self._run_task,
                        agent_command,
                        agent_flags or ["--auto-approve"],
                        index,
                        prompt,
                        worktree,
                        patch_dir,
                        base_commit,
                    ): (index, prompt, worktree, branch)
                    for index, prompt, worktree, branch in specs
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        index, prompt, worktree, _ = futures[future]
                        results.append(
                            ParallelTaskResult(
                                index, prompt, "failed", str(worktree), "", stderr=str(exc), returncode=1
                            )
                        )

            for result in sorted(results, key=lambda item: item.index):
                if result.status != "completed" or not result.patch_path:
                    continue
                patch = Path(result.patch_path)
                if patch.stat().st_size == 0:
                    result.applied = True
                    continue
                check = subprocess.run(
                    ["git", "apply", "--check", str(patch)],
                    cwd=self.project.root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if check.returncode != 0:
                    result.status = "conflict"
                    result.stderr = (result.stderr + "\n" + check.stderr).strip()
                    continue
                apply = subprocess.run(
                    ["git", "apply", str(patch)],
                    cwd=self.project.root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                result.applied = apply.returncode == 0
                if not result.applied:
                    result.status = "conflict"
                    result.stderr = (result.stderr + "\n" + apply.stderr).strip()
            report_path = patch_dir / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "created_at": utc_now_iso(),
                        "task_graph": [
                            {
                                "id": f"task-{index}",
                                "title": prompt,
                                "status": next(
                                    (item.status for item in results if item.index == index),
                                    "failed",
                                ),
                                "dependencies": [],
                                "retry_count": 0,
                                "max_retries": 0,
                                "allow_parallel": True,
                                "completion_criteria": "Agent subprocess exits successfully and patch applies cleanly.",
                            }
                            for index, prompt in enumerate(tasks, start=1)
                        ],
                        "results": [asdict(item) for item in sorted(results, key=lambda item: item.index)],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return run_id, sorted(results, key=lambda item: item.index)
        finally:
            self._cleanup_worktrees(specs, run_dir)

    def _run_task(
        self,
        agent_command: str,
        flags: list[str],
        index: int,
        prompt: str,
        worktree: Path,
        patch_dir: Path,
        base_commit: str,
    ) -> ParallelTaskResult:
        completed = subprocess.run(
            [agent_command, *flags, "--", prompt],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
            env=self._task_environment(index, prompt),
        )
        subprocess.run(
            ["git", "add", "-N", ".", ":(exclude).project-agent"],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        patch = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                base_commit,
                "--",
                ".",
                ":(exclude).project-agent",
            ],
            cwd=worktree,
            text=False,
            capture_output=True,
            check=False,
        )
        patch_path = patch_dir / f"task-{index}.patch"
        patch_path.write_bytes(patch.stdout)
        return ParallelTaskResult(
            index=index,
            prompt=prompt,
            status="completed" if completed.returncode == 0 else "failed",
            worktree=str(worktree),
            patch_path=str(patch_path),
            stdout=completed.stdout[-20_000:],
            stderr=completed.stderr[-20_000:],
            returncode=completed.returncode,
        )

    def _task_environment(self, index: int, prompt: str) -> dict[str, str]:
        plan = [
            {
                "id": f"task-{index}",
                "title": prompt,
                "description": "Git worktree parallel task",
                "dependencies": [],
                "status": "in_progress",
                "retry_count": 0,
                "max_retries": 0,
                "allow_parallel": True,
                "completion_criteria": "Agent subprocess exits successfully and patch applies cleanly.",
            }
        ]
        return {**os.environ, "DEEP_AGENT_INITIAL_PLAN_JSON": json.dumps(plan, ensure_ascii=False)}

    def _require_clean_git(self) -> None:
        if not (self.project.root / ".git").exists():
            raise ValueError("parallel execution requires a Git repository")
        status = self._git("status", "--porcelain", "--", ".", ":(exclude).project-agent")
        if status.strip():
            raise ValueError("parallel execution requires a clean Git working tree")

    def _cleanup_worktrees(self, specs: list[tuple[int, str, Path, str]], run_dir: Path) -> None:
        for _, _, worktree, branch in specs:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=self.project.root,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=self.project.root,
                capture_output=True,
                check=False,
            )
        shutil.rmtree(run_dir, ignore_errors=True)

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.project.root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
        return completed.stdout
