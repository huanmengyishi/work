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
            normalized.append(PlanStep(id=step_id, title=title[:500], status=status))

        state.plan = normalized[:50]
        self._refresh_derived_state(state)
        state.touch()
        return state.plan

    def update_step(self, state: AgentState, step_id: str, status: str) -> PlanStep:
        if status not in VALID_STEP_STATUSES:
            raise ValueError(f"invalid plan status: {status}")
        for step in state.plan:
            if step.id == step_id:
                step.status = status
                self._refresh_derived_state(state)
                state.touch()
                return step
        raise ValueError(f"unknown plan step: {step_id}")

    @staticmethod
    def _step_id(value: str, index: int) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
        return (cleaned or f"step-{index}")[:80]

    @staticmethod
    def _refresh_derived_state(state: AgentState) -> None:
        state.completed_steps = [step.id for step in state.plan if step.status == "completed"]
        active = next((step.id for step in state.plan if step.status == "in_progress"), None)
        if active is None:
            active = next((step.id for step in state.plan if step.status == "pending"), None)
        state.current_step = active
