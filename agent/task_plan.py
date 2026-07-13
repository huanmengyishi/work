from __future__ import annotations

from typing import Any

from .task_router import TaskRoute


class TaskPlanFactory:
    """Build deterministic starter plans without classifying the task.

    TaskRouter owns task type, scale, risk, and mode decisions.  This factory
    consumes the resulting route and only chooses a bounded plan template.
    """

    _CHANGE_TASK_TYPES = frozenset({"bug_fix", "feature_development", "refactor"})

    def build(self, route: TaskRoute) -> list[dict[str, Any]]:
        if not isinstance(route, TaskRoute):
            raise TypeError("TaskPlanFactory requires a TaskRoute from TaskRouter")
        if not route.require_plan:
            return []
        change_task = route.task_type in self._CHANGE_TASK_TYPES or "mutation-request" in route.reasons
        middle_title = "Implement bounded changes" if change_task else "Synthesize the inspected evidence"
        middle_done = (
            "Requested changes are applied through the managed file workflow."
            if change_task
            else "Findings are reconciled across all inspected chunks without unsupported claims."
        )
        middle_id = "implement" if change_task else "synthesize"
        return [
            {
                "id": "scope",
                "title": "Map the request, constraints, and relevant project areas",
                "status": "in_progress",
                "max_retries": 1,
                "completion_criteria": "Scope, constraints, and bounded inspection targets are explicit.",
            },
            {
                "id": "inspect-chunks",
                "title": "Inspect relevant text or code in bounded chunks",
                "dependencies": ["scope"],
                "max_retries": 2,
                "allow_parallel": route.mode == "deep",
                "completion_criteria": "Each relevant chunk has evidence and unresolved questions recorded.",
            },
            {
                "id": middle_id,
                "title": middle_title,
                "dependencies": ["inspect-chunks"],
                "max_retries": 2,
                "completion_criteria": middle_done,
            },
            {
                "id": "verify",
                "title": "Verify the result and reconcile it with the original request",
                "dependencies": [middle_id],
                "max_retries": 1,
                "completion_criteria": "Checks pass and the final answer states evidence, limits, and remaining risk.",
            },
        ]
