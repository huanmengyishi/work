from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping

from .artifact import (
    ARTIFACT_VERIFICATION_METADATA_KEY,
    MAX_ARTIFACT_BYTES_HARD_LIMIT,
    ArtifactSpec,
    ArtifactVerifier,
)
from .constants import MAX_ARTIFACT_REGISTRY_ITEMS
from .memory_refinement import redact_sensitive_text
from .timeutil import utc_now_iso

if TYPE_CHECKING:
    from .state import AgentState


ARTIFACT_STATES = frozenset({"planned", "in_progress", "generated", "verified", "failed"})
_VERIFIABLE_SUFFIXES = frozenset({".docx", ".json", ".yaml", ".yml", ".py", ".pyw", ".md", ".txt", ".rst"})
_LIFECYCLE_CAPABILITIES = frozenset(
    {
        ("document", "parse"),
        ("document", "render_docx"),
        ("document_generator", "create_outline"),
        ("document_generator", "confirm_outline"),
        ("document_generator", "finalize"),
        ("document_generator", "next_chapter"),
        ("document_generator", "render"),
        ("document_generator", "rollback_chapter"),
        ("document_generator", "save_chapter"),
        ("document_generator", "status"),
        ("file", "apply"),
        ("file", "diff"),
        ("file", "undo"),
        ("template", "make_dir"),
    }
)


class ArtifactRegistry:
    """Maintain bounded artifact lifecycle facts from managed ToolResults only."""

    SCHEMA_VERSION = 1

    @classmethod
    def sync_planned(cls, state: AgentState) -> None:
        registry = cls._registry(state)
        route_hints = list((state.task_route or {}).get("artifact_hints") or [])
        for raw in list(dict.fromkeys(str(item).strip() for item in route_hints if str(item).strip()))[:64]:
            key = cls._normalize_path(state, raw) or f"hint:{raw[:200]}"
            artifacts = registry["artifacts"]
            if key not in artifacts and len(artifacts) < MAX_ARTIFACT_REGISTRY_ITEMS:
                artifacts[key] = cls._entry(path=raw, state="planned", kind=cls._kind(raw))
        by_step = {step.id: step for step in state.plan}
        for step in state.plan[:128]:
            parent = by_step.get(step.parent_id or "")
            parent_artifacts = [
                cls._normalize_path(state, value) or f"hint:{str(value)[:200]}"
                for value in (parent.artifact_ids if parent is not None else [])[:32]
            ]
            for raw in list(dict.fromkeys(str(item).strip() for item in step.artifact_ids if str(item).strip()))[:32]:
                key = cls._normalize_path(state, raw) or f"hint:{raw[:200]}"
                artifacts = registry["artifacts"]
                if key not in artifacts:
                    if len(artifacts) >= MAX_ARTIFACT_REGISTRY_ITEMS:
                        continue
                    artifacts[key] = cls._entry(path=raw, state="planned", kind=cls._kind(raw))
                entry = artifacts[key]
                cls._append_unique(entry, "step_ids", step.id, maximum=32)
                if step.parent_id:
                    cls._append_unique(entry, "parent_step_ids", step.parent_id, maximum=32)
                for parent_artifact in parent_artifacts:
                    if parent_artifact != key:
                        cls._append_unique(entry, "parent_artifacts", parent_artifact, maximum=32)

    @classmethod
    def observe_state(cls, state: AgentState, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        registry = cls._registry(state)
        artifacts: dict[str, dict[str, Any]] = registry["artifacts"]
        tool = str(request.get("tool") or "")
        action = str(request.get("action") or "")
        args = request.get("args") if isinstance(request.get("args"), Mapping) else {}
        data = result.get("data") if isinstance(result.get("data"), Mapping) else {}
        success = result.get("success") is True and not bool(data.get("not_executed"))
        capability = (tool, action)

        if capability == ("file", "undo"):
            if success:
                snapshot_id = str(data.get("snapshot_id") or args.get("snapshot_id") or "")
                restored_exists = data.get("restored_exists") is True
                for entry in artifacts.values():
                    if snapshot_id and entry.get("snapshot_id") == snapshot_id:
                        entry.update(
                            {
                                "state": "generated" if restored_exists else "planned",
                                "generated": restored_exists,
                                "verified": False,
                                "snapshot_id": "",
                                "updated_at": utc_now_iso(),
                                "last_transition": "undo",
                            }
                        )
            return

        # A path-bearing read, diagnostic, shell command, or unknown capability
        # is not an artifact lifecycle transition.  In particular, a failed
        # read must never erase truthful evidence from an earlier managed write.
        if capability not in _LIFECYCLE_CAPABILITIES:
            if success:
                cls._record_source_dates(registry, data)
            return

        path_value = data.get("path") or data.get("output_path") or args.get("path") or args.get("output_path")
        path = cls._normalize_path(state, str(path_value or ""))
        if not path:
            if success:
                cls._record_source_dates(registry, data)
            return

        # Verification consumes an existing planned/generated lineage. Parsing
        # an arbitrary source document must not fill the artifact registry.
        entry = cls._existing_entry_for_path(state, artifacts, path)
        if entry is None and capability == ("document", "parse"):
            if success:
                cls._record_source_dates(registry, data)
            return
        if entry is None:
            entry = cls._entry_for_path(state, artifacts, path)
        capability_name = f"{tool}.{action}"
        entry["last_capability"] = capability_name[:160]
        entry["updated_at"] = utc_now_iso()
        entry["turn"] = state.turn
        entry["round"] = state.round
        if state.current_step:
            cls._append_unique(entry, "step_ids", state.current_step, maximum=32)
        if not success:
            # A failed verification invalidates a verification claim. A failed
            # preview/write attempt cannot undo a previously generated file or
            # directory because its handler did not report a successful effect.
            if capability == ("document", "parse") or entry.get("state") not in {"generated", "verified"}:
                entry["state"] = "failed"
                entry["verified"] = False
            entry["error"] = redact_sensitive_text(str(result.get("stderr") or "tool failed"), maximum=500)
            return

        entry["error"] = ""
        if capability == ("template", "make_dir"):
            entry.update({"kind": "directory", "state": "generated", "generated": True})
        elif tool == "document_generator":
            workflow_status = str(data.get("workflow_status") or "")[:64]
            workflow_event = str(data.get("document_workflow_event") or "")[:64]
            completed_chapters = cls._bounded_count(data.get("completed_chapters"))
            total_chapters = cls._bounded_count(data.get("total_chapters"))
            entry.update(
                {
                    "workflow_id": str(data.get("workflow_id") or "")[:200],
                    "workflow_status": workflow_status,
                    "workflow_event": workflow_event,
                    "completed_chapters": completed_chapters,
                    "total_chapters": total_chapters,
                }
            )
            for key, maximum in (
                ("outline_hash", 64),
                ("chapter_id", 64),
                ("chapter_sha256", 64),
            ):
                if data.get(key):
                    entry[key] = str(data.get(key) or "")[:maximum]
            if action == "create_outline":
                entry.update({"state": "planned", "generated": False, "workflow_finalized": False})
            elif action == "render":
                preview_id = str(data.get("preview_id") or "")[:200]
                entry.update(
                    {
                        "state": "in_progress",
                        "generated": False,
                        "verified": False,
                        "preview_id": preview_id,
                        "workflow_render_preview_id": preview_id,
                        "workflow_apply_matches": False,
                        "workflow_finalized": False,
                        "generated_metadata_dates": [
                            str(item)[:80] for item in list(data.get("generated_metadata_dates") or [])[:100]
                        ],
                    }
                )
            elif action == "finalize":
                expected_preview = str(entry.get("workflow_render_preview_id") or "")
                finalized_preview = str(data.get("render_preview_id") or "")
                apply_verified = data.get("apply_verified") is True
                if apply_verified and expected_preview and finalized_preview == expected_preview:
                    entry.update(
                        {
                            "state": "verified" if entry.get("verified") else "generated",
                            "generated": True,
                            "workflow_apply_matches": True,
                            "workflow_finalized": True,
                            "snapshot_id": str(data.get("apply_snapshot_id") or entry.get("snapshot_id") or "")[:200],
                            "result_hash": str(data.get("apply_result_hash") or "")[:64],
                        }
                    )
                else:
                    entry.update(
                        {
                            "state": "in_progress",
                            "generated": False,
                            "verified": False,
                            "workflow_finalized": False,
                            "error": "document workflow finalize receipt did not match its render preview",
                        }
                    )
            elif entry.get("state") not in {"generated", "verified"}:
                entry["state"] = "planned" if workflow_status == "awaiting_confirmation" else "in_progress"
        elif capability in {
            ("file", "diff"),
            ("document", "render_docx"),
        }:
            entry.update(
                {
                    "state": "in_progress",
                    "preview_id": str(data.get("preview_id") or "")[:200],
                    "workflow_id": str(data.get("workflow_id") or entry.get("workflow_id") or "")[:200],
                    "generated_metadata_dates": [
                        str(item)[:80] for item in list(data.get("generated_metadata_dates") or [])[:100]
                    ],
                }
            )
        elif capability == ("file", "apply"):
            after_exists = data.get("after_exists")
            route_schema = (state.task_route or {}).get("schema_version", 1)
            legacy_unknown_exists = (
                after_exists is None
                and isinstance(route_schema, int)
                and not isinstance(route_schema, bool)
                and route_schema < 2
            )
            generated = after_exists is True or legacy_unknown_exists
            applied_preview = str(data.get("preview_id") or "")[:200]
            workflow_preview = str(entry.get("workflow_render_preview_id") or "")
            workflow_matches = not workflow_preview or applied_preview == workflow_preview
            if generated and workflow_matches:
                entry.update(
                    {
                        "state": "generated",
                        "generated": True,
                        "verified": False,
                        "preview_id": applied_preview,
                        "snapshot_id": str(data.get("snapshot_id") or "")[:200],
                        "result_hash": str(data.get("result_hash") or "")[:64],
                        "workflow_apply_matches": bool(workflow_preview),
                    }
                )
            else:
                entry.update(
                    {
                        "state": "in_progress" if workflow_preview else "planned",
                        "generated": False,
                        "verified": False,
                        "workflow_apply_matches": False,
                        "snapshot_id": str(data.get("snapshot_id") or "")[:200],
                        "last_applied_preview_id": applied_preview,
                    }
                )
                if workflow_preview and applied_preview != workflow_preview:
                    entry["error"] = "applied preview does not match the active document workflow render"
        elif capability == ("document", "parse"):
            receipt = data.get(ARTIFACT_VERIFICATION_METADATA_KEY)
            passed = cls._receipt_passes(path, result, receipt)
            if passed and entry.get("generated"):
                entry.update({"state": "verified", "verified": True, "verification": cls._safe_receipt(receipt)})
            elif receipt is not None:
                entry.update({"state": "failed", "verified": False, "error": "artifact receipt failed"})
        cls._record_source_dates(registry, data)

    @classmethod
    def completion_issue(cls, state: AgentState, reasons: set[str]) -> tuple[bool, str]:
        """Return ``(handled, issue)`` without weakening the legacy gate."""

        value = state.artifact_registry
        artifacts = value.get("artifacts") if isinstance(value, dict) else None
        if not isinstance(artifacts, dict) or not artifacts:
            return False, ""
        if "artifact-required" not in reasons:
            return True, ""
        cls.sync_planned(state)
        route_hints = [str(item) for item in (state.task_route or {}).get("artifact_hints", []) if str(item)]
        directory_hints = [str(item) for item in (state.task_route or {}).get("directory_hints", []) if str(item)]
        if (
            not route_hints
            and not directory_hints
            and not any(
                item.get("state") in {"generated", "verified"} for item in artifacts.values() if isinstance(item, dict)
            )
        ):
            return True, (
                "the requested output artifact is not generated in ArtifactRegistry; "
                "there is no active successful managed-write evidence"
            )
        for hint in directory_hints:
            entry = cls._matching_entry(state, artifacts, hint)
            if not entry or entry.get("kind") != "directory" or entry.get("state") not in {"generated", "verified"}:
                return True, (
                    "the requested output directory is not generated in ArtifactRegistry; "
                    f"there is no successful managed make_dir evidence matching: {hint}"
                )
        for hint in route_hints:
            if hint in directory_hints or ("." not in PurePosixPath(hint).name and not Path(hint).suffix):
                continue
            entry = cls._matching_entry(state, artifacts, hint)
            if not entry or entry.get("state") not in {"generated", "verified"}:
                return True, (
                    "the requested output artifact is not generated in ArtifactRegistry; "
                    f"there is no active successful managed-write evidence matching: {hint}"
                )
            if hint.lower().endswith(".docx") and entry.get("state") != "verified":
                return True, f"the requested Word artifact is not verified in ArtifactRegistry: {hint}"
            if entry.get("workflow_id") and entry.get("workflow_finalized") is not True:
                return True, f"the requested document workflow is not finalized in ArtifactRegistry: {hint}"
        return True, ""

    @classmethod
    def _registry(cls, state: AgentState) -> dict[str, Any]:
        value = state.artifact_registry if isinstance(state.artifact_registry, dict) else {}
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        value.update({"schema_version": cls.SCHEMA_VERSION, "artifacts": artifacts})
        value.setdefault("source_date_literals", [])
        state.artifact_registry = value
        return value

    @classmethod
    def _entry_for_path(
        cls,
        state: AgentState,
        artifacts: dict[str, dict[str, Any]],
        path: str,
    ) -> dict[str, Any]:
        existing = artifacts.get(path)
        if isinstance(existing, dict):
            return existing
        for key, value in list(artifacts.items()):
            if key.startswith("hint:") and cls._hint_matches(key[5:], path):
                artifacts.pop(key)
                value["path"] = path
                value["kind"] = cls._kind(path)
                artifacts[path] = value
                return value
        if len(artifacts) >= MAX_ARTIFACT_REGISTRY_ITEMS:
            # Prefer dropping an unresolved planned hint over active lineage.
            removable = next((key for key, item in artifacts.items() if item.get("state") == "planned"), None)
            if removable is not None:
                artifacts.pop(removable)
            else:
                return cls._entry(path=path, state="failed", kind=cls._kind(path))
        entry = cls._entry(path=path, state="planned", kind=cls._kind(path))
        artifacts[path] = entry
        return entry

    @classmethod
    def _existing_entry_for_path(
        cls,
        state: AgentState,
        artifacts: dict[str, dict[str, Any]],
        path: str,
    ) -> dict[str, Any] | None:
        existing = artifacts.get(path)
        if isinstance(existing, dict):
            return existing
        for key, value in list(artifacts.items()):
            if key.startswith("hint:") and cls._hint_matches(key[5:], path) and isinstance(value, dict):
                artifacts.pop(key)
                value["path"] = path
                value["kind"] = cls._kind(path)
                artifacts[path] = value
                return value
        return None

    @classmethod
    def _matching_entry(
        cls,
        state: AgentState,
        artifacts: Mapping[str, dict[str, Any]],
        hint: str,
    ) -> dict[str, Any] | None:
        normalized = cls._normalize_path(state, hint)
        if normalized and isinstance(artifacts.get(normalized), dict):
            return artifacts[normalized]
        matches = [value for key, value in artifacts.items() if cls._hint_matches(hint, key)]
        return (
            max(matches, key=lambda item: (int(item.get("turn") or 0), int(item.get("round") or 0)))
            if matches
            else None
        )

    @staticmethod
    def _entry(*, path: str, state: str, kind: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "path": path[:500],
            "kind": kind,
            "state": state if state in ARTIFACT_STATES else "planned",
            "generated": False,
            "verified": False,
            "preview_id": "",
            "snapshot_id": "",
            "error": "",
            "turn": 0,
            "round": 0,
            "step_ids": [],
            "parent_step_ids": [],
            "parent_artifacts": [],
            "updated_at": utc_now_iso(),
        }

    @staticmethod
    def _append_unique(entry: dict[str, Any], key: str, value: str, *, maximum: int) -> None:
        normalized = str(value).strip()[:200]
        if not normalized:
            return
        existing = entry.get(key)
        values = [str(item)[:200] for item in existing if str(item).strip()] if isinstance(existing, list) else []
        if normalized not in values and len(values) < maximum:
            values.append(normalized)
        entry[key] = values[:maximum]

    @staticmethod
    def _kind(path: str) -> str:
        return "directory" if not PurePosixPath(str(path)).suffix else "file"

    @staticmethod
    def _normalize_path(state: AgentState, value: str) -> str:
        raw = str(value).strip().replace("\\", "/")
        if not raw or raw.startswith("hint:"):
            return ""
        path = Path(raw)
        root = Path(state.working_directory).resolve(strict=False)
        resolved = path.resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return ""

    @staticmethod
    def _hint_matches(hint: str, path: str) -> bool:
        normalized_hint = str(hint).strip().replace("\\", "/").rstrip("/")
        normalized_path = str(path).strip().replace("\\", "/").rstrip("/")
        if not normalized_hint or not normalized_path:
            return False
        basename = normalized_path.rsplit("/", 1)[-1]
        return (
            basename.lower().endswith(normalized_hint.lower())
            if normalized_hint.startswith(".")
            else basename == normalized_hint
        )

    @staticmethod
    def _receipt_passes(path: str, result: Mapping[str, Any], receipt: object) -> bool:
        if not isinstance(receipt, Mapping):
            return False
        try:
            spec = ArtifactSpec(
                str(receipt.get("artifact_id") or "document-parse"),
                path,
                format=str(receipt.get("format") or "auto"),
                max_bytes=MAX_ARTIFACT_BYTES_HARD_LIMIT,
            )
        except (TypeError, ValueError):
            return False
        return ArtifactVerifier.verify_receipt(spec, result).passed

    @staticmethod
    def _safe_receipt(value: object) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        allowed = {
            "schema_version",
            "artifact_id",
            "path",
            "format",
            "passed",
            "content_complete",
            "size_bytes",
            "content_sha256",
            "checks_run",
            "errors",
        }
        return {str(key): item for key, item in value.items() if key in allowed}

    @staticmethod
    def _record_source_dates(registry: dict[str, Any], data: Mapping[str, Any]) -> None:
        existing = [str(item)[:80] for item in registry.get("source_date_literals", []) if str(item)]
        for item in list(data.get("date_literals") or [])[:100]:
            text = str(item)[:80]
            if text and text not in existing:
                existing.append(text)
        registry["source_date_literals"] = existing[:200]

    @staticmethod
    def _bounded_count(value: object) -> int:
        return max(0, min(value, 1_000_000)) if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = ["ARTIFACT_STATES", "ArtifactRegistry"]
