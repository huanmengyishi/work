from __future__ import annotations

import re
from typing import Any

from .state import AgentState, PlanStep


VALID_STEP_STATUSES = {"pending", "in_progress", "completed", "failed", "skipped"}


class PlanManager:
    """Validates model-generated plans and keeps derived state consistent."""

    def replace(self, state: AgentState, steps: list[str | dict[str, Any]]) -> list[PlanStep]:
        normalized: list[PlanStep] = []
        for index, value in enumerate(steps, start=1):
            if isinstance(value, str):
                title = value.strip()
                status = "pending"
                requested_id = ""
            elif isinstance(value, dict):
                title = str(value.get("title") or value.get("step") or "").strip()
                status = str(value.get("status") or "pending")
                requested_id = str(value.get("id") or "")
            else:
                continue
            if not title:
                continue
            if status not in VALID_STEP_STATUSES:
                status = "pending"
            step_id = self._step_id(requested_id or f"step-{index}", index)
            dependencies = value.get("dependencies", []) if isinstance(value, dict) else []
            normalized.append(
                PlanStep(
                    id=step_id,
                    title=title[:500],
                    status=status,
                    description=str(value.get("description") or "")[:2000] if isinstance(value, dict) else "",
                    dependencies=[str(item)[:80] for item in dependencies] if isinstance(dependencies, list) else [],
                    retry_count=max(0, int(value.get("retry_count") or 0)) if isinstance(value, dict) else 0,
                    max_retries=min(10, max(0, int(value.get("max_retries") or 0))) if isinstance(value, dict) else 0,
                    allow_parallel=bool(value.get("allow_parallel", False)) if isinstance(value, dict) else False,
                    completion_criteria=str(value.get("completion_criteria") or "")[:2000]
                    if isinstance(value, dict)
                    else "",
                )
            )

        normalized = normalized[:50]
        self._validate_graph(normalized)
        state.plan = normalized
        self._refresh_derived_state(state)
        state.touch()
        return state.plan

    def update_step(self, state: AgentState, step_id: str, status: str) -> PlanStep:
        if status not in VALID_STEP_STATUSES:
            raise ValueError(f"invalid plan status: {status}")
        for step in state.plan:
            if step.id == step_id:
                if status == "in_progress" and not self.dependencies_satisfied(state, step):
                    raise ValueError(f"plan step dependencies are not complete: {step_id}")
                if status == "failed" and step.retry_count < step.max_retries:
                    step.retry_count += 1
                    step.status = "pending"
                    self._refresh_derived_state(state)
                    state.touch()
                    return step
                step.status = status
                self._refresh_derived_state(state)
                state.touch()
                return step
        raise ValueError(f"unknown plan step: {step_id}")

    @staticmethod
    def dependencies_satisfied(state: AgentState, step: PlanStep) -> bool:
        completed = {item.id for item in state.plan if item.status in {"completed", "skipped"}}
        return all(dependency in completed for dependency in step.dependencies)

    def ready_steps(self, state: AgentState) -> list[PlanStep]:
        return [step for step in state.plan if step.status == "pending" and self.dependencies_satisfied(state, step)]

    @staticmethod
    def _step_id(value: str, index: int) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
        return (cleaned or f"step-{index}")[:80]

    @staticmethod
    def _refresh_derived_state(state: AgentState) -> None:
        state.completed_steps = [step.id for step in state.plan if step.status == "completed"]
        active = next((step.id for step in state.plan if step.status == "in_progress"), None)
        if active is None:
            completed = {step.id for step in state.plan if step.status in {"completed", "skipped"}}
            active = next(
                (
                    step.id
                    for step in state.plan
                    if step.status == "pending" and all(dependency in completed for dependency in step.dependencies)
                ),
                None,
            )
        state.current_step = active
        if state.execution_context:
            state.execution_context.current_plan_id = active

    @staticmethod
    def _validate_graph(steps: list[PlanStep]) -> None:
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step IDs must be unique")
        known = set(ids)
        for step in steps:
            missing = [item for item in step.dependencies if item not in known]
            if missing:
                raise ValueError(f"unknown dependencies for {step.id}: {', '.join(missing)}")
            if step.id in step.dependencies:
                raise ValueError(f"plan step cannot depend on itself: {step.id}")
            if step.status == "in_progress" and step.dependencies:
                completed = {item.id for item in steps if item.status in {"completed", "skipped"}}
                if not all(item in completed for item in step.dependencies):
                    raise ValueError(f"in-progress step dependencies are not complete: {step.id}")
        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {step.id: step.dependencies for step in steps}

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("plan dependencies contain a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in graph[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in graph:
            visit(step_id)
