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
        if route.task_type == "document_workflow":
            steps: list[dict[str, Any]] = [
                {
                    "id": "scope",
                    "title": "Discover the requested documents, exclusions, and output path",
                    "status": "in_progress",
                    "max_retries": 1,
                    "completion_criteria": "The bounded input set and requested artifact are explicit.",
                },
                {
                    "id": "parse-documents",
                    "title": "Parse every selected document in bounded batches",
                    "dependencies": ["scope"],
                    "max_retries": 2,
                    "allow_parallel": True,
                    "completion_criteria": "Each selected document has a successful parse result or a reported error.",
                },
                {
                    "id": "synthesize",
                    "title": "Synthesize the requested summary from the parsed evidence",
                    "dependencies": ["parse-documents"],
                    "max_retries": 2,
                    "completion_criteria": "The summary covers the selected sources without unsupported claims.",
                },
            ]
            if "artifact-required" in route.reasons:
                steps.extend(
                    [
                        {
                            "id": "render-artifact",
                            "title": "Create the requested document through a managed snapshot-backed tool",
                            "dependencies": ["synthesize"],
                            "max_retries": 2,
                            "completion_criteria": (
                                "The requested output artifact exists through the managed write workflow."
                            ),
                        },
                        {
                            "id": "verify",
                            "title": "Re-open and verify the generated document",
                            "dependencies": ["render-artifact"],
                            "max_retries": 1,
                            "completion_criteria": "The artifact parses successfully and contains the requested summary.",
                        },
                    ]
                )
            else:
                steps.append(
                    {
                        "id": "verify",
                        "title": "Verify the summary against the parsed sources",
                        "dependencies": ["synthesize"],
                        "max_retries": 1,
                        "completion_criteria": "The final summary covers the requested sources and states any limits.",
                    }
                )
            return steps
        change_task = route.task_type in self._CHANGE_TASK_TYPES or "mutation-request" in route.reasons
        middle_title = "Implement bounded changes" if change_task else "Synthesize the inspected evidence"
        middle_done = (
            "Requested changes are applied through the managed file workflow."
            if change_task
            else "Findings are reconciled across all inspected chunks without unsupported claims."
        )
        if change_task and "conditional-mutation" in route.reasons:
            middle_done = (
                "A proven issue is changed through the managed file workflow, or implementation is skipped with "
                "explicit evidence that no justified mutation was found."
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
                "completion_criteria": (
                    "Relevant checks are executed and their exact outcomes are reported. A pass is claimed only when "
                    "the checks pass; pre-existing failures or environment limitations are recorded with evidence. "
                    "The final answer states limits and remaining risk."
                ),
            },
        ]
