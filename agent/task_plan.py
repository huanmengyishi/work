from __future__ import annotations

from typing import Any, Protocol

from .task_router import TaskRoute


class TaskPlanStrategy(Protocol):
    """One deterministic plan policy selected from an existing TaskRoute."""

    task_types: frozenset[str]

    def matches(self, route: TaskRoute) -> bool: ...

    def build(self, route: TaskRoute) -> list[dict[str, Any]]: ...


class DocumentWorkflowStrategy:
    task_types: frozenset[str] = frozenset({"document_workflow"})

    def matches(self, route: TaskRoute) -> bool:
        return route.task_type in self.task_types

    def build(self, route: TaskRoute) -> list[dict[str, Any]]:
        scale_rounds = _scale_rounds(route)
        steps: list[dict[str, Any]] = [
            _step(
                "scope",
                "Discover the requested documents, exclusions, and output path",
                status="in_progress",
                step_type="scope",
                rounds=1,
                weight=1.0,
                retries=1,
                criteria="The bounded input set and requested artifact are explicit.",
            ),
            _step(
                "parse-documents",
                "Parse every selected document in bounded batches",
                dependencies=["scope"],
                step_type="inspect",
                rounds=scale_rounds + 1,
                weight=3.0,
                retries=2,
                allow_parallel=True,
                criteria="Each selected document has a successful parse result or a reported error.",
            ),
            _step(
                "synthesize",
                "Synthesize the requested summary from the parsed evidence",
                dependencies=["parse-documents"],
                step_type="synthesize",
                rounds=scale_rounds,
                weight=3.0,
                retries=2,
                criteria="The summary covers the selected sources without unsupported claims.",
            ),
        ]
        if "artifact-required" in route.reasons:
            steps.extend(
                [
                    _step(
                        "render-artifact",
                        "Create the requested document through a managed snapshot-backed tool",
                        dependencies=["synthesize"],
                        step_type="render",
                        rounds=2,
                        weight=2.0,
                        retries=2,
                        artifacts=list(route.artifact_hints),
                        criteria="The requested output artifact exists through the managed write workflow.",
                    ),
                    _step(
                        "verify",
                        "Re-open and verify the generated document",
                        dependencies=["render-artifact"],
                        step_type="verify",
                        rounds=1,
                        weight=1.0,
                        retries=1,
                        artifacts=list(route.artifact_hints),
                        validations=["document_parse", "nonempty", "size"],
                        criteria="The artifact parses successfully and contains the requested summary.",
                    ),
                ]
            )
        else:
            steps.append(
                _step(
                    "verify",
                    "Verify the summary against the parsed sources",
                    dependencies=["synthesize"],
                    step_type="verify",
                    rounds=1,
                    weight=1.0,
                    retries=1,
                    validations=["source_coverage"],
                    criteria="The final summary covers the requested sources and states any limits.",
                )
            )
        return steps


class ChangeWorkflowStrategy:
    task_types: frozenset[str] = frozenset()
    title = "Implement bounded changes"

    def matches(self, route: TaskRoute) -> bool:
        return route.task_type in self.task_types

    def build(self, route: TaskRoute) -> list[dict[str, Any]]:
        criteria = "Requested changes are applied through the managed file workflow."
        if "conditional-mutation" in route.reasons:
            criteria = (
                "A proven issue is changed through the managed file workflow, or implementation is skipped with "
                "explicit evidence that no justified mutation was found."
            )
        return _standard_plan(route, middle_id="implement", middle_title=self.title, middle_criteria=criteria)


class BugFixStrategy(ChangeWorkflowStrategy):
    task_types = frozenset({"bug_fix"})
    title = "Implement the proven root-cause fix"


class FeatureDevStrategy(ChangeWorkflowStrategy):
    task_types = frozenset({"feature_development"})
    title = "Implement the bounded feature and acceptance behavior"


class RefactorStrategy(ChangeWorkflowStrategy):
    task_types = frozenset({"refactor"})
    title = "Refactor while preserving observable behavior"


class MutationWorkflowStrategy(ChangeWorkflowStrategy):
    """Catch mutation intent after all task-type-specific strategies."""

    def matches(self, route: TaskRoute) -> bool:
        return route.task_type != "document_workflow" and "mutation-request" in route.reasons


class EvidenceWorkflowStrategy:
    task_types: frozenset[str] = frozenset({"question", "code_explanation", "review", "architecture"})

    def matches(self, route: TaskRoute) -> bool:
        return route.task_type in self.task_types

    def build(self, route: TaskRoute) -> list[dict[str, Any]]:
        titles = {
            "architecture": "Produce the evidence-backed architecture decision",
            "review": "Reconcile review findings by severity and evidence",
        }
        return _standard_plan(
            route,
            middle_id="synthesize",
            middle_title=titles.get(route.task_type, "Synthesize the inspected evidence"),
            middle_criteria="Findings are reconciled across all inspected chunks without unsupported claims.",
        )


class TaskPlanFactory:
    """Select a registered deterministic strategy without reclassifying text."""

    def __init__(self, strategies: tuple[TaskPlanStrategy, ...] | None = None) -> None:
        self.strategies = strategies or (
            DocumentWorkflowStrategy(),
            BugFixStrategy(),
            FeatureDevStrategy(),
            RefactorStrategy(),
            MutationWorkflowStrategy(),
            EvidenceWorkflowStrategy(),
        )

    def build(self, route: TaskRoute) -> list[dict[str, Any]]:
        if not isinstance(route, TaskRoute):
            raise TypeError("TaskPlanFactory requires a TaskRoute from TaskRouter")
        if not route.require_plan:
            return []
        for strategy in self.strategies:
            if strategy.matches(route):
                return strategy.build(route)
        fallback: TaskPlanStrategy = EvidenceWorkflowStrategy()
        return fallback.build(route)


def _standard_plan(
    route: TaskRoute,
    *,
    middle_id: str,
    middle_title: str,
    middle_criteria: str,
) -> list[dict[str, Any]]:
    scale_rounds = _scale_rounds(route)
    change_task = middle_id == "implement"
    return [
        _step(
            "scope",
            "Map the request, constraints, and relevant project areas",
            status="in_progress",
            step_type="scope",
            rounds=1,
            weight=1.0,
            retries=1,
            criteria="Scope, constraints, and bounded inspection targets are explicit.",
        ),
        _step(
            "inspect-chunks",
            "Inspect relevant text or code in bounded chunks",
            dependencies=["scope"],
            step_type="inspect",
            rounds=scale_rounds + int(route.risk == "high"),
            weight=3.0,
            retries=2,
            allow_parallel=route.mode == "deep",
            criteria="Each relevant chunk has evidence and unresolved questions recorded.",
        ),
        _step(
            middle_id,
            middle_title,
            dependencies=["inspect-chunks"],
            step_type="implement" if change_task else "synthesize",
            rounds=scale_rounds + int(change_task),
            weight=4.0 if change_task else 3.0,
            retries=2,
            artifacts=list(route.artifact_hints),
            criteria=middle_criteria,
        ),
        _step(
            "verify",
            "Verify the result and reconcile it with the original request",
            dependencies=[middle_id],
            step_type="verify",
            rounds=1 + int(route.risk == "high"),
            weight=2.0,
            retries=1,
            artifacts=list(route.artifact_hints),
            validations=["managed_validation"] if change_task else ["evidence_reconciliation"],
            criteria=(
                "Relevant checks are executed and their exact outcomes are reported. A pass is claimed only when "
                "the checks pass; pre-existing failures or environment limitations are recorded with evidence. "
                "The final answer states limits and remaining risk."
            ),
        ),
    ]


def _step(
    step_id: str,
    title: str,
    *,
    status: str = "pending",
    dependencies: list[str] | None = None,
    step_type: str,
    rounds: int,
    weight: float,
    retries: int,
    criteria: str,
    allow_parallel: bool = False,
    artifacts: list[str] | None = None,
    validations: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": step_id,
        "title": title,
        "status": status,
        "step_type": step_type,
        "estimated_tool_rounds": rounds,
        "progress_weight": weight,
        "max_retries": retries,
        "completion_criteria": criteria,
    }
    if dependencies:
        value["dependencies"] = dependencies
    if allow_parallel:
        value["allow_parallel"] = True
    if artifacts:
        value["artifact_ids"] = artifacts
    if validations:
        value["validation_rules"] = validations
    return value


def _scale_rounds(route: TaskRoute) -> int:
    return {"small": 1, "medium": 2, "large": 3}.get(route.scale, 2)


__all__ = [
    "BugFixStrategy",
    "DocumentWorkflowStrategy",
    "EvidenceWorkflowStrategy",
    "FeatureDevStrategy",
    "MutationWorkflowStrategy",
    "RefactorStrategy",
    "TaskPlanFactory",
    "TaskPlanStrategy",
]
