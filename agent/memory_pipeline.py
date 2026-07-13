from __future__ import annotations

import re
from typing import Any

from .config import AppConfig
from .events import Event, EventBus
from .memory import MemoryStore
from .project import Project
from .reflection import ReflectionEngine


class MemoryPipeline:
    """Converts completed task evidence into searchable, structured memory."""

    def __init__(
        self,
        *,
        config: AppConfig,
        project: Project,
        memory: MemoryStore,
        events: EventBus,
    ) -> None:
        self.config = config
        self.project = project
        self.memory = memory
        self.events = events
        self.reflection = ReflectionEngine(config)
        events.subscribe("task.finished", self.handle)
        events.subscribe("task.failed", self.handle)

    def handle(self, event: Event) -> None:
        state = event.payload.get("state") or {}
        run_id = str(state.get("run_id") or event.payload.get("run_id") or "")
        if run_id and self.memory.is_pipeline_run_processed(run_id):
            return

        prompt = str(event.payload.get("prompt") or state.get("user_request") or "").strip()
        final = str(event.payload.get("final") or state.get("final_answer") or "").strip()
        error = str(event.payload.get("error") or state.get("error") or "").strip()
        session_id = str(state.get("session_id") or event.session_id or "unknown")
        turn = int(state.get("turn") or 1)
        tool_calls = [item for item in list(state.get("tool_calls") or []) if int(item.get("turn") or 1) == turn]
        success = event.name == "task.finished"

        summary_id: int | None = None
        if self.config.get("runtime.auto_summarize", True):
            summary = self._summary(prompt, final, error, tool_calls, success)
            summary_id = self.memory.add_memory(
                kind="Summary",
                title=f"Session {session_id} turn {turn}",
                content=summary,
                tags=["session", f"turn-{turn}", self.project.language.lower()],
                project_id=self.project.id,
            )
            self.memory.update_summary(scope="latest_session", content=summary, project_id=self.project.id)
            self.events.publish(
                "memory.summary.persisted",
                {"memory_id": summary_id, "run_id": run_id},
                project_id=self.project.id,
                session_id=session_id,
            )

        experience_id: int | None = None
        if self.config.get("runtime.write_lessons", True) and tool_calls:
            kind = self._classify(prompt, error, success)
            experience = self._experience(prompt, final, error, tool_calls, success)
            tags = [kind.lower(), "automatic", self.project.language.lower()]
            experience_id = self.memory.add_memory(
                kind=kind,
                title=f"{kind}: {self._title(prompt)}",
                content=experience,
                tags=tags,
                project_id=self.project.id,
            )
            self.memory.persist_lesson_file(
                kind=kind,
                title=f"{kind}: {self._title(prompt)}",
                content=experience,
                project=self.project,
                global_memory=False,
            )
            self.events.publish(
                "memory.experience.persisted",
                {"memory_id": experience_id, "kind": kind, "run_id": run_id},
                project_id=self.project.id,
                session_id=session_id,
            )

        reflection = self.reflection.reflect(
            prompt=prompt,
            final=final,
            error=error,
            tool_calls=tool_calls,
            success=success,
        )
        if reflection:
            reflection_id = self.memory.add_memory(
                kind="Reflection",
                title=f"Reflection: {self._title(prompt)}",
                content=reflection,
                tags=["reflection", "automatic", "success" if success else "failed"],
                project_id=self.project.id,
            )
            self.memory.persist_lesson_file(
                kind="Reflection",
                title=f"Reflection: {self._title(prompt)}",
                content=reflection,
                project=self.project,
                global_memory=False,
            )
            self.events.publish(
                "memory.reflection.persisted",
                {"memory_id": reflection_id, "run_id": run_id},
                project_id=self.project.id,
                session_id=session_id,
            )

        if run_id:
            self.memory.mark_pipeline_run_processed(run_id, self.project.id, summary_id, experience_id)

    @staticmethod
    def _classify(prompt: str, error: str, success: bool) -> str:
        text = f"{prompt}\n{error}".lower()
        if not success or re.search(r"\b(bug|fix|error|exception|failure)\b|修复|错误|异常|故障", text):
            return "Bug"
        if re.search(r"\b(architecture|design|decision|refactor|migration)\b|架构|设计|决策|选型|重构|迁移", text):
            return "Decision"
        return "Lesson"

    @staticmethod
    def _summary(prompt: str, final: str, error: str, tool_calls: list[dict[str, Any]], success: bool) -> str:
        outcome = final or error or "No final output was recorded."
        return "\n".join(
            [
                f"Status: {'completed' if success else 'failed'}",
                f"Request: {prompt[:3000]}",
                f"Tool calls: {len(tool_calls)}",
                "Outcome:",
                outcome[:5000],
            ]
        )

    @staticmethod
    def _experience(
        prompt: str,
        final: str,
        error: str,
        tool_calls: list[dict[str, Any]],
        success: bool,
    ) -> str:
        failures = []
        for item in tool_calls:
            request = item.get("request") or {}
            result = item.get("result") or {}
            if not result.get("success"):
                failures.append(
                    f"{request.get('tool', '?')}.{request.get('action', '?')}: "
                    f"{str(result.get('stderr') or 'failed')[:500]}"
                )
        evidence = "\n".join(failures[:10]) or f"完成了 {len(tool_calls)} 次受管工具调用。"
        return "\n".join(
            [
                "问题",
                prompt[:3000],
                "",
                "原因与证据",
                evidence,
                "",
                "解决",
                (final or error or "任务未生成最终说明。")[:5000],
                "",
                "影响",
                "任务结果、工具证据和分类已写入项目记忆，可供后续 SQLite/Chroma 检索。",
                "",
                "标签",
                f"automatic, {'success' if success else 'failed'}",
            ]
        )

    @staticmethod
    def _title(prompt: str) -> str:
        title = " ".join(prompt.split())[:100]
        return title or "Untitled task"
