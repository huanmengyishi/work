from __future__ import annotations

import re
from typing import Any, Mapping

from .config import AppConfig
from .events import Event, EventBus
from .memory import MemoryStore
from .memory_refinement import MemoryRefiner, redact_sensitive_text, sanitize_memory_tags
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
        run_id = str(state.get("run_id") or event.effective_run_id or "")
        if run_id and self.memory.is_pipeline_run_processed(run_id):
            return

        # Session state is private recovery data, but derived long-term Memory
        # is searched and reused across future prompts.  Redact every textual
        # source before it can enter any Summary/Experience/Reflection record.
        prompt = redact_sensitive_text(
            str(event.payload.get("prompt") or state.get("user_request") or "").strip(),
            maximum=32_000,
        )
        final = redact_sensitive_text(
            str(event.payload.get("final") or state.get("final_answer") or "").strip(),
            maximum=50_000,
        )
        error = redact_sensitive_text(
            str(event.payload.get("error") or state.get("error") or "").strip(),
            maximum=10_000,
        )
        session_id = str(state.get("session_id") or event.session_id or "unknown")
        turn = int(state.get("turn") or 1)
        tool_calls = MemoryRefiner.current_turn_tool_calls(list(state.get("tool_calls") or []), turn=turn)
        success = event.name == "task.finished"
        refinement = self._accepted_refinement(event.payload.get("memory_refinement"), tool_calls, success)

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
                run_id=run_id or None,
            )

        experience_id: int | None = None
        if self.config.get("runtime.write_lessons", True) and tool_calls:
            kind = str(refinement.get("kind")) if refinement else self._classify(prompt, error, success)
            experience = (
                str(refinement.get("experience") or "")
                if refinement
                else self._experience(prompt, final, error, tool_calls, success)
            )
            tags = [kind.lower(), "automatic", self.project.language.lower()]
            if refinement:
                tags.extend(str(item) for item in refinement.get("tags", []))
                tags = list(dict.fromkeys(tags))[:20]
            refined_title = str(refinement.get("title") or "") if refinement else ""
            experience_id = self.memory.add_memory(
                kind=kind,
                title=f"{kind}: {refined_title or self._title(prompt)}"[:200],
                content=experience,
                tags=tags,
                project_id=self.project.id,
                confidence=float(refinement["confidence"]) if refinement else None,
            )
            self.memory.persist_lesson_file(
                kind=kind,
                title=f"{kind}: {refined_title or self._title(prompt)}"[:200],
                content=experience,
                project=self.project,
                global_memory=False,
            )
            self.events.publish(
                "memory.experience.persisted",
                {"memory_id": experience_id, "kind": kind, "run_id": run_id},
                project_id=self.project.id,
                session_id=session_id,
                run_id=run_id or None,
            )

        reflection = self.reflection.reflect(
            prompt=prompt,
            final=final,
            error=error,
            tool_calls=tool_calls,
            success=success,
            smart_text=str(refinement.get("reflection") or "") if refinement else None,
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
                run_id=run_id or None,
            )

        if run_id:
            self.memory.mark_pipeline_run_processed(run_id, self.project.id, summary_id, experience_id)

    def _accepted_refinement(
        self,
        value: object,
        tool_calls: list[dict[str, Any]],
        success: bool,
    ) -> dict[str, Any] | None:
        """Defensively accept only Runtime-validated refinement payloads."""

        eligible, _reason = self.reflection_eligibility(success=success, tool_call_count=len(tool_calls))
        if not eligible or not isinstance(value, Mapping) or len(value) > 16:
            return None
        title = redact_sensitive_text(str(value.get("title") or "").strip(), maximum=160)
        experience = redact_sensitive_text(str(value.get("experience") or "").strip(), maximum=5_000)
        reflection = redact_sensitive_text(str(value.get("reflection") or "").strip(), maximum=2_000)
        # Treat the event payload as untrusted even though Runtime already
        # validated it. This second pass prevents a forged terminal event from
        # smuggling a credential through searchable tag metadata.
        tags = sanitize_memory_tags(value.get("tags"))
        kind = str(value.get("kind") or "").strip().title()
        if kind not in {"Lesson", "Bug", "Decision"}:
            return None
        try:
            confidence = float(value.get("confidence"))
        except (TypeError, ValueError, OverflowError):
            return None
        if not 0.0 <= confidence <= 1.0 or not title or not experience or not reflection:
            return None
        return {
            "kind": kind,
            "title": title,
            "experience": experience,
            "reflection": reflection,
            "tags": list(tags),
            "confidence": confidence,
        }

    def reflection_eligibility(self, *, success: bool, tool_call_count: int) -> tuple[bool, str]:
        return MemoryRefiner(self.config).eligible(success=success, current_tool_calls=tool_call_count)

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
                detail = redact_sensitive_text(str(result.get("stderr") or "failed"), maximum=500)
                failures.append(f"{request.get('tool', '?')}.{request.get('action', '?')}: {detail}")
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
