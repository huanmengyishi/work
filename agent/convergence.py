from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .convergence_guards import (
    BROAD_EXPLORATION_FUNCTIONS as _BROAD_EXPLORATION_FUNCTIONS,
    MAX_IMPLEMENTATION_READ_LINES as _MAX_IMPLEMENTATION_READ_LINES,
    MAX_PERSISTED_SEEN_TARGETS as _MAX_PERSISTED_SEEN_TARGETS,
    MAX_TARGET_KEY_CHARS as _MAX_TARGET_KEY_CHARS,
    MAX_VALIDATION_ATTACHMENT_READ_CHARS as _MAX_VALIDATION_ATTACHMENT_READ_CHARS,
    READ_ONLY_CAPABILITIES as _READ_ONLY_CAPABILITIES,
    TARGETED_EXPLORATION_FUNCTIONS as _TARGETED_EXPLORATION_FUNCTIONS,
    bounded_turn,
    conditional_mutation_step_active,
    exploration_step_active,
    has_validation_attachment,
    implementation_or_verification_step_active,
    implementation_read_denial,
    implementation_step_active,
    is_bounded_validation_command,
    is_exploration_bypass,
    is_validation_attachment,
    normalized_path,
    path_was_read_successfully,
    plan_requires_transition,
    step_semantic_type,
    target_key,
    validation_attachment_id,
    validation_attachment_read_denial,
)
from .context_window import (
    ContextWindowController,
    PairRepairResult,
    RequestTokenBudget,
    estimate_request_tokens,
    repair_tool_message_pairs,
)
from .history_compaction import ToolHistoryCompactor, ToolHistoryResult
from .state import AgentState


@dataclass(frozen=True)
class ConvergenceAction:
    messages: tuple[str, ...] = ()
    excluded_functions: frozenset[str] = frozenset()
    reason: str = ""
    block_exploration_bypass: bool = False
    guard_implementation_read: bool = False
    guard_validation_attachment_read: bool = False
    force_plan_transition: bool = False


class TaskConvergenceController:
    """Detect low-yield exploration and preserve implementation/verify rounds."""

    def __init__(
        self,
        *,
        mode: str,
        max_rounds: int,
        exploration_round_limit: int,
        reserved_rounds: int,
        implementation_read_limit: int = 2,
        validation_attachment_read_limit: int = 2,
    ) -> None:
        self.enabled = mode in {"large", "deep"}
        self.max_rounds = max(1, int(max_rounds))
        self.exploration_round_limit = max(2, min(int(exploration_round_limit), self.max_rounds))
        self.reserved_rounds = max(1, min(int(reserved_rounds), max(1, self.max_rounds - 1)))
        self.implementation_read_limit = max(0, min(int(implementation_read_limit), 4))
        self.implementation_reads_used = 0
        self.validation_attachment_read_limit = max(0, min(int(validation_attachment_read_limit), 4))
        self.validation_attachment_reads_used = 0
        self.consecutive_read_only_rounds = 0
        self.low_yield_rounds = 0
        self.seen_targets: set[str] = set()
        self.last_plan_fingerprint: tuple[tuple[str, str], ...] = ()
        self.nudge_count = 0
        self.exploration_rounds_observed = 0
        self._nudge_sent_for_stall = False
        self._hard_notice_sent = False
        self._implementation_notice_sent = False
        self._last_implementation_notice_remaining: int | None = None
        self._last_validation_attachment_notice_remaining: int | None = None
        self._bound_state: AgentState | None = None
        self._seen_target_order: list[str] = []

    def bind(self, state: AgentState) -> None:
        self._bound_state = state
        self.last_plan_fingerprint = self._plan_fingerprint(state)
        metadata = getattr(state, "convergence", {})
        current_turn = self._bounded_turn(state)
        if isinstance(metadata, dict):
            used = metadata.get("implementation_reads_used", 0)
            if isinstance(used, int) and not isinstance(used, bool):
                self.implementation_reads_used = max(0, min(used, 4))
            validation_reads = metadata.get("validation_attachment_reads_used", 0)
            if isinstance(validation_reads, int) and not isinstance(validation_reads, bool):
                self.validation_attachment_reads_used = max(0, min(validation_reads, 4))
            consecutive = metadata.get("consecutive_read_only_rounds", 0)
            if isinstance(consecutive, int) and not isinstance(consecutive, bool):
                self.consecutive_read_only_rounds = max(
                    0,
                    min(consecutive, self.exploration_round_limit + 2),
                )
            low_yield = metadata.get("low_yield_rounds", 0)
            if isinstance(low_yield, int) and not isinstance(low_yield, bool):
                self.low_yield_rounds = max(0, min(low_yield, 5))
            raw_targets = metadata.get("seen_targets", [])
            if isinstance(raw_targets, list):
                targets = [
                    item
                    for item in raw_targets[-_MAX_PERSISTED_SEEN_TARGETS:]
                    if isinstance(item, str) and 0 < len(item) <= _MAX_TARGET_KEY_CHARS
                ]
                self._seen_target_order = list(dict.fromkeys(targets))
                self.seen_targets = set(self._seen_target_order)
            nudge_count = metadata.get("nudge_count", 0)
            exploration_rounds = metadata.get("exploration_rounds_observed", 0)
            if isinstance(exploration_rounds, int) and not isinstance(exploration_rounds, bool):
                self.exploration_rounds_observed = max(0, min(exploration_rounds, 10_000))
            same_turn = metadata.get("notice_turn") == current_turn
            if same_turn:
                if isinstance(nudge_count, int) and not isinstance(nudge_count, bool):
                    self.nudge_count = max(0, min(nudge_count, 2))
                self._nudge_sent_for_stall = metadata.get("nudge_sent_for_stall") is True
                self._hard_notice_sent = metadata.get("hard_notice_sent") is True
        self._sync()

    def _sync(self) -> None:
        if self._bound_state is None:
            return
        metadata = getattr(self._bound_state, "convergence", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self._bound_state, "convergence", metadata)
        metadata.update(
            {
                "implementation_reads_used": self.implementation_reads_used,
                "validation_attachment_reads_used": self.validation_attachment_reads_used,
                "consecutive_read_only_rounds": self.consecutive_read_only_rounds,
                "low_yield_rounds": self.low_yield_rounds,
                "seen_targets": list(self._seen_target_order[-_MAX_PERSISTED_SEEN_TARGETS:]),
                "nudge_count": self.nudge_count,
                "exploration_rounds_observed": self.exploration_rounds_observed,
                "nudge_sent_for_stall": self._nudge_sent_for_stall,
                "hard_notice_sent": self._hard_notice_sent,
                "notice_turn": self._bounded_turn(self._bound_state),
            }
        )

    @staticmethod
    def _bounded_turn(state: Any) -> int:
        return bounded_turn(state)

    def before_round(self, round_number: int, state: Any | None = None) -> ConvergenceAction:
        if not self.enabled:
            return ConvergenceAction()
        reserve_due = round_number > self.max_rounds - self.reserved_rounds
        stalled = self.consecutive_read_only_rounds >= self.exploration_round_limit or self.low_yield_rounds >= 3
        forced = self.consecutive_read_only_rounds >= self.exploration_round_limit + 2 or self.low_yield_rounds >= 5
        hard = reserve_due or forced
        force_plan_transition = hard and (self._exploration_step_active(state) or self._plan_requires_transition(state))
        implementation_read_open = hard and self._implementation_step_active(state) and self._read_allowance_remaining()
        validation_attachment_read_open = (
            hard
            and self._implementation_or_verification_step_active(state)
            and self._validation_attachment_allowance_remaining()
            and self._has_validation_attachment(state)
        )
        if not stalled and not reserve_due:
            return ConvergenceAction()

        notices: list[str] = []
        implementation_remaining = max(0, self.implementation_read_limit - self.implementation_reads_used)
        remaining = self.max_rounds - round_number + 1
        if not self._nudge_sent_for_stall and self.nudge_count < 2:
            notices.append(
                "Exploration budget checkpoint: the task has spent "
                f"{self.consecutive_read_only_rounds} consecutive rounds on read-only or non-progress inspection "
                "without advancing "
                "the Task Graph. Stop broad scanning, update the current plan step, and synthesize the evidence "
                f"already collected. Preserve the remaining {remaining} tool rounds for a concrete change when "
                "justified, static checks, verification, and the final answer. Read only an exact missing target "
                "that is necessary for implementation; do not use shell or Python to bypass this checkpoint."
            )
            self._nudge_sent_for_stall = True
            self.nudge_count += 1
        if hard and not self._hard_notice_sent:
            notice = (
                "The exploration window is now closed because its continuous-read threshold or reserved-round "
                "boundary was reached. Use the existing evidence, plan updates, managed edits, diagnostics/tests, "
                "or return a substantive evidence-based answer. Shell or Python file-reading commands are also "
                "blocked in this phase; they must not be used to bypass the managed exploration tools."
            )
            if implementation_read_open:
                notice += (
                    " Because the implement step is active, read_file remains available only for an exact path "
                    f"already read successfully, with explicit start/end lines covering at most "
                    f"{_MAX_IMPLEMENTATION_READ_LINES} lines. "
                    f"At most {self.implementation_read_limit - self.implementation_reads_used} such evidence "
                    "read(s) remain; broad or new-target inspection is still closed."
                )
            elif force_plan_transition:
                notice += (
                    " The current scope/inspection step must transition now. In this response, use "
                    "agent_update_step to complete the current exploration step and start the next ready step; "
                    "do not spend another round on status, tests, diagnostics, or file inspection."
                )
            if self._conditional_mutation_step_active(state):
                notice += (
                    " This is a conditional-mutation plan. If the evidence already collected does not prove a real "
                    "issue that justifies a change, do not invent one: call agent_update_step with step_id "
                    "`implement` and status `skipped`, then start `verify` and report the exact validation outcome."
                )
            notices.append(notice)
            self._hard_notice_sent = True
            if implementation_read_open:
                self._implementation_notice_sent = True
                self._last_implementation_notice_remaining = implementation_remaining
        elif implementation_read_open and (
            not self._implementation_notice_sent
            or self._last_implementation_notice_remaining != implementation_remaining
        ):
            notices.append(
                "The implement step is now active inside the closed exploration window. read_file is available "
                "only as a bounded implementation-evidence exception: use an exact path that was read "
                "successfully before the window closed, provide explicit start_line/end_line values covering at "
                f"most {_MAX_IMPLEMENTATION_READ_LINES} lines. Bounded implementation evidence allowance: "
                f"{self.implementation_read_limit - self.implementation_reads_used} read(s) remaining. "
                "New targets, broad reads, shell/Python file exploration, and verify-phase reads remain closed."
            )
            self._implementation_notice_sent = True
            self._last_implementation_notice_remaining = implementation_remaining
        elif (
            hard
            and self._implementation_step_active(state)
            and self._last_implementation_notice_remaining != implementation_remaining
        ):
            notices.append(
                f"Bounded implementation evidence allowance: {implementation_remaining} read(s) remaining. "
                + (
                    "read_file is now closed; proceed with the managed edit, verification, or final answer."
                    if implementation_remaining == 0
                    else "Only an exact previously-read path and a range of at most 200 lines is allowed."
                )
            )
            self._last_implementation_notice_remaining = implementation_remaining

        validation_attachment_remaining = max(
            0,
            self.validation_attachment_read_limit - self.validation_attachment_reads_used,
        )
        if (
            validation_attachment_read_open
            and self._last_validation_attachment_notice_remaining != validation_attachment_remaining
        ):
            notices.append(
                "A bounded validation attachment is available in the closed exploration window. tool_result_read "
                "may read only an attachment produced by run_tests, diagnostics, document verification, or staged-"
                f"diff verification in this Session; each chunk is limited to {_MAX_VALIDATION_ATTACHMENT_READ_CHARS} "
                f"characters and {validation_attachment_remaining} "
                "read(s) remain. It cannot read ordinary exploration attachments."
            )
            self._last_validation_attachment_notice_remaining = validation_attachment_remaining

        excluded = set(_BROAD_EXPLORATION_FUNCTIONS)
        reason = "read-only exploration stalled"
        if hard:
            excluded.update(_TARGETED_EXPLORATION_FUNCTIONS)
            if implementation_read_open:
                excluded.discard("read_file")
            if validation_attachment_read_open:
                excluded.discard("tool_result_read")
            reason = (
                "reserved implementation and verification window"
                if reserve_due
                else "continuous exploration threshold reached"
            )
        action = ConvergenceAction(
            tuple(notices),
            frozenset(excluded),
            reason,
            block_exploration_bypass=hard,
            guard_implementation_read=implementation_read_open,
            guard_validation_attachment_read=validation_attachment_read_open,
            force_plan_transition=force_plan_transition,
        )
        self._sync()
        return action

    def implementation_read_denial(
        self,
        state: Any,
        function_name: str,
        arguments: str | dict[str, Any] | None,
    ) -> str:
        """Consume one narrowly scoped implementation evidence read or explain denial."""

        denial = implementation_read_denial(
            state,
            function_name,
            arguments,
            allowance_remaining=self._read_allowance_remaining(),
        )
        if denial:
            return denial
        self.implementation_reads_used += 1
        self._implementation_notice_sent = False
        if self._bound_state is None and hasattr(state, "convergence"):
            self._bound_state = state
        self._sync()
        return ""

    def _read_allowance_remaining(self) -> bool:
        return self.implementation_reads_used < self.implementation_read_limit

    def validation_attachment_read_denial(
        self,
        state: Any,
        function_name: str,
        arguments: str | dict[str, Any] | None,
    ) -> str:
        """Consume one bounded read of a validation-produced private attachment."""

        denial = validation_attachment_read_denial(
            state,
            function_name,
            arguments,
            allowance_remaining=self._validation_attachment_allowance_remaining(),
        )
        if denial:
            return denial
        self.validation_attachment_reads_used += 1
        if self._bound_state is None and hasattr(state, "convergence"):
            self._bound_state = state
        self._sync()
        return ""

    def _validation_attachment_allowance_remaining(self) -> bool:
        return self.validation_attachment_reads_used < self.validation_attachment_read_limit

    @classmethod
    def _has_validation_attachment(cls, state: Any | None) -> bool:
        return has_validation_attachment(state)

    @classmethod
    def _is_validation_attachment(cls, state: Any, request_id: str) -> bool:
        return is_validation_attachment(state, request_id)

    @classmethod
    def _validation_attachment_id(cls, item: Any) -> str:
        return validation_attachment_id(item)

    @classmethod
    def _implementation_step_active(cls, state: Any | None) -> bool:
        return implementation_step_active(state)

    @classmethod
    def _implementation_or_verification_step_active(cls, state: Any | None) -> bool:
        return implementation_or_verification_step_active(state)

    @classmethod
    def _conditional_mutation_step_active(cls, state: Any | None) -> bool:
        return conditional_mutation_step_active(state)

    @classmethod
    def _exploration_step_active(cls, state: Any | None) -> bool:
        return exploration_step_active(state)

    @staticmethod
    def _step_semantic_type(step: Any) -> str:
        return step_semantic_type(step)

    @staticmethod
    def _plan_requires_transition(state: Any | None) -> bool:
        return plan_requires_transition(state)

    @staticmethod
    def _normalized_path(value: Any) -> str:
        return normalized_path(value)

    @classmethod
    def _path_was_read_successfully(cls, state: Any, path: str) -> bool:
        return path_was_read_successfully(state, path)

    def observe_round(
        self,
        state: AgentState,
        requests: list[dict[str, Any]],
        results: list[dict[str, Any]] | None = None,
    ) -> bool:
        plan_fingerprint = self._plan_fingerprint(state)
        plan_progressed = plan_fingerprint != self.last_plan_fingerprint
        self.last_plan_fingerprint = plan_fingerprint
        capabilities = [f"{item.get('tool', '')}.{item.get('action', '')}" for item in requests]
        read_only = bool(capabilities) and all(item in _READ_ONLY_CAPABILITIES for item in capabilities)
        if self.enabled and read_only:
            self.exploration_rounds_observed = min(10_000, self.exploration_rounds_observed + 1)
        targets = [self._target_key(item) for item in requests if self._target_key(item)]
        repeated_targets = sum(target in self.seen_targets for target in targets)
        new_targets = len(targets) - repeated_targets
        for target in targets:
            if target in self.seen_targets:
                continue
            self.seen_targets.add(target)
            self._seen_target_order.append(target)
        while len(self._seen_target_order) > _MAX_PERSISTED_SEEN_TARGETS:
            removed = self._seen_target_order.pop(0)
            self.seen_targets.discard(removed)

        result_items = results or []
        productive_capabilities = {
            "file.apply",
            "file.undo",
            "document.render_docx",
            "template.run_tests",
            "lsp.diagnostics",
        }
        productive = any(
            capability in productive_capabilities
            and index < len(result_items)
            and bool(result_items[index].get("success"))
            for index, capability in enumerate(capabilities)
        )
        progressed = plan_progressed or productive
        if not self.enabled:
            self._sync()
            return progressed
        if progressed:
            self.consecutive_read_only_rounds = 0
            self.low_yield_rounds = 0
            self._nudge_sent_for_stall = False
            self._hard_notice_sent = False
            self._sync()
            return True
        if requests:
            self.consecutive_read_only_rounds = min(
                self.exploration_round_limit + 2,
                self.consecutive_read_only_rounds + 1,
            )
        if read_only and targets and repeated_targets >= max(1, new_targets):
            self.low_yield_rounds = min(5, self.low_yield_rounds + 1)
        else:
            self.low_yield_rounds = max(0, self.low_yield_rounds - 1)
        self._sync()
        return False

    @staticmethod
    def filter_schemas(schemas: list[dict[str, Any]], excluded_functions: frozenset[str]) -> list[dict[str, Any]]:
        if not excluded_functions:
            return schemas
        return [
            item for item in schemas if str((item.get("function") or {}).get("name") or "") not in excluded_functions
        ]

    @staticmethod
    def _plan_fingerprint(state: AgentState) -> tuple[tuple[str, str], ...]:
        return tuple((step.id, step.status) for step in state.plan)

    @staticmethod
    def _target_key(request: dict[str, Any]) -> str:
        return target_key(request)

    @staticmethod
    def is_exploration_bypass(function_name: str, arguments: str | dict[str, Any] | None) -> bool:
        return is_exploration_bypass(function_name, arguments)

    @staticmethod
    def _is_bounded_validation_command(command: str) -> bool:
        return is_bounded_validation_command(command)


__all__ = [
    "ConvergenceAction",
    "ContextWindowController",
    "PairRepairResult",
    "RequestTokenBudget",
    "TaskConvergenceController",
    "ToolHistoryCompactor",
    "ToolHistoryResult",
    "estimate_request_tokens",
    "repair_tool_message_pairs",
]
