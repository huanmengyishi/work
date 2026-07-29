from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .config import AppConfig
from .events import Event, EventBus
from .paths import storage_key
from .timeutil import utc_now_iso


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MODEL_TIERS = frozenset({"standard", "deep"})


@dataclass(frozen=True)
class ExperimentAssignment:
    experiment_id: str
    variant: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    """Default-off deterministic A/B allocation with scalar-only outcomes."""

    MAX_EXPERIMENTS = 16
    MAX_VARIANTS = 8
    MAX_RESULT_BYTES = 4 * 1024 * 1024

    def __init__(self, config: AppConfig, *, project_id: str) -> None:
        self.enabled = bool(config.get("optimizer.experiments.enabled", False))
        self.salt = str(config.get("optimizer.experiments.salt", "deep-agent"))[:128]
        raw = config.get("optimizer.experiments.definitions", [])
        self.definitions = self._definitions(raw)
        self.path = config.data_dir / "experiments" / f"{storage_key(project_id)}.jsonl"
        self.project_id = project_id

    def attach(self, events: EventBus) -> None:
        events.subscribe("task.finished", self.handle, name="experiments.outcome")
        events.subscribe("task.failed", self.handle, name="experiments.outcome")

    def assign(self, *, run_id: str, task_type: str) -> ExperimentAssignment | None:
        if not self.enabled:
            return None
        for definition in self.definitions:
            task_types = definition["task_types"]
            if task_types and task_type not in task_types:
                continue
            variants = definition["variants"]
            total = sum(item["weight"] for item in variants)
            digest = hashlib.sha256(
                f"{self.salt}\0{self.project_id}\0{run_id}\0{definition['id']}".encode("utf-8")
            ).digest()
            slot = int.from_bytes(digest[:8], "big") % total
            cumulative = 0
            for variant in variants:
                cumulative += variant["weight"]
                if slot < cumulative:
                    return ExperimentAssignment(
                        definition["id"],
                        variant["name"],
                        dict(variant["parameters"]),
                    )
        return None

    def handle(self, event: Event) -> None:
        state = event.payload.get("state")
        if not self.enabled or not isinstance(state, Mapping):
            return
        convergence = state.get("convergence")
        assignment = convergence.get("experiment") if isinstance(convergence, Mapping) else None
        if not isinstance(assignment, Mapping) or not assignment.get("experiment_id"):
            return
        record = {
            "schema_version": 1,
            "run_id": str(event.effective_run_id or "")[:200],
            "project_id": str(event.project_id or self.project_id)[:128],
            "experiment_id": str(assignment.get("experiment_id") or "")[:64],
            "variant": str(assignment.get("variant") or "")[:64],
            "outcome": "completed" if event.name == "task.finished" else "failed",
            "recorded_at": utc_now_iso(),
        }
        try:
            self._append(record)
        except (OSError, ValueError):
            return

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise OSError("experiment directory must not be a symbolic link")
        if self.path.exists() or self.path.is_symlink():
            metadata = self.path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > self.MAX_RESULT_BYTES:
                raise OSError("experiment result path is invalid or oversized")
        content = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if self.path.exists() and self.path.stat().st_size + len(content) > self.MAX_RESULT_BYTES:
            return
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(content)
        try:
            self.path.chmod(0o600)
            self.path.parent.chmod(0o700)
        except OSError:
            pass

    @classmethod
    def _definitions(cls, value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        definitions: list[dict[str, Any]] = []
        for raw in value[: cls.MAX_EXPERIMENTS]:
            if not isinstance(raw, Mapping) or raw.get("enabled", True) is not True:
                continue
            experiment_id = str(raw.get("id") or "")
            raw_variants = raw.get("variants")
            if not _IDENTIFIER.fullmatch(experiment_id) or not isinstance(raw_variants, list):
                continue
            variants: list[dict[str, Any]] = []
            for item in raw_variants[: cls.MAX_VARIANTS]:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name") or "")
                weight = item.get("weight", 1)
                if not _IDENTIFIER.fullmatch(name) or isinstance(weight, bool) or not isinstance(weight, int):
                    continue
                parameters = cls._parameters(item.get("parameters"))
                variants.append({"name": name, "weight": max(1, min(weight, 10_000)), "parameters": parameters})
            if len(variants) < 2:
                continue
            task_types = raw.get("task_types")
            definitions.append(
                {
                    "id": experiment_id,
                    "task_types": tuple(str(item)[:64] for item in task_types[:16])
                    if isinstance(task_types, list)
                    else (),
                    "variants": variants,
                }
            )
        return definitions

    @staticmethod
    def _parameters(value: object) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        result: dict[str, Any] = {}
        exploration = value.get("exploration_round_limit")
        if isinstance(exploration, int) and not isinstance(exploration, bool):
            result["exploration_round_limit"] = max(2, min(exploration, 32))
        tier = str(value.get("model_tier") or "")
        if tier in _MODEL_TIERS:
            result["model_tier"] = tier
        return result


__all__ = ["ExperimentAssignment", "ExperimentRunner"]
