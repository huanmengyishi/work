from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from agent.artifact import (
    ARTIFACT_VERIFICATION_METADATA_KEY,
    MANAGED_DOCUMENT_ARTIFACT_ID,
    ArtifactSpec,
    ArtifactVerifier,
    artifact_verification_metadata,
    verify_artifact,
)
from agent.artifact_registry import ArtifactRegistry
from agent.progress import ProgressTracker, derive_progress
from agent.runtime_validation import RuntimeValidationMixin
from agent.state import AgentState, PlanStep
from agent.tools.base import ToolResult


def _state(*, status: str = "running", plan: list[PlanStep] | None = None) -> AgentState:
    steps = plan or []
    current = next((step.id for step in steps if step.status == "in_progress"), None)
    return AgentState(
        session_id="progress-test",
        project={"id": "project", "name": "project", "root": "/workspace"},
        objective="test progress",
        user_request="test progress",
        request_history=["test progress"],
        working_directory="/workspace",
        status=status,
        plan=steps,
        current_step=current,
        completed_steps=[step.id for step in steps if step.status == "completed"],
    )


def _docx_bytes(text: str, *, extra: bytes = b"") -> bytes:
    target = BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships '
            'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="word/document.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>'
            "</Relationships>",
        )
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>'
            + text
            + "</w:t></w:r></w:p></w:body></w:document>",
        )
        if extra:
            archive.writestr("word/extra.bin", extra)
    return target.getvalue()


def test_artifact_registry_preserves_generated_verified_and_undo_lineage_beyond_hot_window() -> None:
    state = _state()
    state.task_route = {
        "artifact_hints": ["report.docx"],
        "directory_hints": [],
        "reasons": ["artifact-required", "word-artifact-required"],
    }
    ArtifactRegistry.sync_planned(state)
    state.record_tool_call(
        {"tool": "document", "action": "render_docx", "args": {"path": "report.docx"}},
        {"success": True, "data": {"path": "report.docx", "preview_id": "preview-1"}},
    )
    state.record_tool_call(
        {"tool": "file", "action": "apply", "args": {"preview_id": "preview-1"}},
        {
            "success": True,
            "data": {
                "path": "report.docx",
                "preview_id": "preview-1",
                "snapshot_id": "snapshot-1",
                "after_exists": True,
            },
        },
    )
    payload = _docx_bytes("verified registry")
    spec = ArtifactSpec(MANAGED_DOCUMENT_ARTIFACT_ID, "report.docx", format="docx")
    verification = verify_artifact(
        spec,
        ToolResult(True, "", data={"exists": True, "size_bytes": len(payload), "content_complete": True}),
        content=payload,
    )
    receipt = artifact_verification_metadata(spec, verification, content=payload)
    state.record_tool_call(
        {"tool": "document", "action": "parse", "args": {"path": "report.docx"}},
        {
            "success": True,
            "data": {"path": "report.docx", ARTIFACT_VERIFICATION_METADATA_KEY: receipt},
        },
    )

    for index in range(205):
        state.record_tool_call(
            {"tool": "template", "action": "read_file", "args": {"path": f"src/{index}.py"}},
            {"success": True, "data": {}},
        )

    entry = state.artifact_registry["artifacts"]["report.docx"]
    assert entry["state"] == "verified"
    assert ArtifactRegistry.completion_issue(state, {"artifact-required", "word-artifact-required"}) == (True, "")
    assert state.tool_history_summary["count"] > 0

    state.record_tool_call(
        {"tool": "file", "action": "undo", "args": {"snapshot_id": "snapshot-1"}},
        {
            "success": True,
            "data": {"path": "report.docx", "snapshot_id": "snapshot-1", "restored_exists": False},
        },
    )

    assert entry["state"] == "planned"
    handled, issue = ArtifactRegistry.completion_issue(
        state,
        {"artifact-required", "word-artifact-required"},
    )
    assert handled is True
    assert "not generated" in issue


def test_artifact_registry_records_cross_step_parent_lineage() -> None:
    state = _state(
        plan=[
            PlanStep(
                "bundle",
                "Prepare output bundle",
                status="in_progress",
                step_type="generate",
                artifact_ids=["out"],
            ),
            PlanStep(
                "report",
                "Generate report",
                parent_id="bundle",
                step_type="render",
                artifact_ids=["out/report.docx"],
            ),
        ]
    )

    ArtifactRegistry.sync_planned(state)

    parent = state.artifact_registry["artifacts"]["out"]
    child = state.artifact_registry["artifacts"]["out/report.docx"]
    assert parent["step_ids"] == ["bundle"]
    assert child["step_ids"] == ["report"]
    assert child["parent_step_ids"] == ["bundle"]
    assert child["parent_artifacts"] == ["out"]


def test_artifact_registry_requires_schema2_exists_evidence_and_tracks_undo_restore() -> None:
    state = _state()
    state.task_route = {
        "schema_version": 2,
        "artifact_hints": ["report.txt"],
        "directory_hints": [],
        "reasons": ["artifact-required"],
    }
    ArtifactRegistry.sync_planned(state)

    state.record_tool_call(
        {"tool": "file", "action": "apply", "args": {"preview_id": "missing-exists"}},
        {
            "success": True,
            "data": {
                "path": "report.txt",
                "preview_id": "missing-exists",
                "snapshot_id": "missing-exists-snapshot",
            },
        },
    )
    entry = state.artifact_registry["artifacts"]["report.txt"]
    assert entry["state"] == "planned"

    state.record_tool_call(
        {"tool": "file", "action": "apply", "args": {"preview_id": "delete-preview"}},
        {
            "success": True,
            "data": {
                "path": "report.txt",
                "preview_id": "delete-preview",
                "snapshot_id": "delete-snapshot",
                "after_exists": False,
            },
        },
    )
    assert entry["state"] == "planned"

    state.record_tool_call(
        {"tool": "file", "action": "undo", "args": {"snapshot_id": "delete-snapshot"}},
        {
            "success": True,
            "data": {
                "path": "report.txt",
                "snapshot_id": "delete-snapshot",
                "restored_exists": True,
            },
        },
    )
    assert entry["state"] == "generated"
    assert entry["generated"] is True
    assert entry["verified"] is False
    assert RuntimeValidationMixin._artifact_evidence_issue(state, {"artifact-required"}) == ""


def test_artifact_registry_ignores_failed_reads_and_gate_is_stable_after_hot_window_pruning() -> None:
    state = _state()
    state.task_route = {
        "schema_version": 2,
        "artifact_hints": ["report.txt"],
        "directory_hints": [],
        "reasons": ["artifact-required"],
    }
    ArtifactRegistry.sync_planned(state)
    state.record_tool_call(
        {"tool": "file", "action": "apply", "args": {"preview_id": "preview-1"}},
        {
            "success": True,
            "data": {
                "path": "report.txt",
                "preview_id": "preview-1",
                "snapshot_id": "snapshot-1",
                "after_exists": True,
            },
        },
    )
    state.record_tool_call(
        {"tool": "template", "action": "read_file", "args": {"path": "report.txt"}},
        {
            "success": False,
            "stderr": "transient read failure",
            "data": {"date_literals": ["2099-01-01"]},
        },
    )

    entry = state.artifact_registry["artifacts"]["report.txt"]
    assert entry["state"] == "generated"
    assert set(state.artifact_registry["artifacts"]) == {"report.txt"}
    assert state.artifact_registry["source_date_literals"] == []
    issue_before_pruning = RuntimeValidationMixin._artifact_evidence_issue(state, {"artifact-required"})
    assert issue_before_pruning == ""

    state.record_tool_call(
        {"tool": "template", "action": "read_file", "args": {"path": "unknown.txt"}},
        {"success": True, "data": {"path": "unknown.txt"}},
    )
    state.record_tool_call(
        {"tool": "file", "action": "apply", "args": {"preview_id": "outside-preview"}},
        {
            "success": True,
            "data": {
                "path": str(Path(state.working_directory).parent / "outside.txt"),
                "preview_id": "outside-preview",
                "snapshot_id": "outside-snapshot",
                "after_exists": True,
            },
        },
    )

    assert entry["state"] == "generated"
    assert set(state.artifact_registry["artifacts"]) == {"report.txt"}

    for index in range(205):
        state.record_tool_call(
            {"tool": "template", "action": "read_file", "args": {"path": f"src/{index}.py"}},
            {"success": True, "data": {}},
        )

    assert state.tool_history_summary["count"] > 0
    assert entry["state"] == "generated"
    assert set(state.artifact_registry["artifacts"]) == {"report.txt"}
    assert RuntimeValidationMixin._artifact_evidence_issue(state, {"artifact-required"}) == issue_before_pruning


def test_progress_counts_only_state_accepted_completed_and_skipped_steps() -> None:
    valid = _state(
        plan=[
            PlanStep("scope", "Scope", status="completed"),
            PlanStep("implement", "Implement", status="skipped", dependencies=["scope"]),
        ]
    )
    valid.task_route = {"reasons": ["conditional-mutation"]}
    snapshot = derive_progress(valid)
    assert snapshot.percent == 99.0
    assert snapshot.completed_steps == 1
    assert snapshot.skipped_steps == 1

    invalid = _state(plan=list(valid.plan))
    invalid_snapshot = derive_progress(invalid)
    assert invalid_snapshot.percent == 50.0
    assert invalid_snapshot.skipped_steps == 0


def test_progress_empty_plan_current_step_and_real_budget_usage() -> None:
    assert ProgressTracker.snapshot(_state()).percent == 0.0
    assert ProgressTracker.snapshot(_state(status="completed")).percent == 100.0

    state = _state(
        plan=[
            PlanStep("scope", "Scope", status="completed"),
            PlanStep("verify", "Verify", status="in_progress", dependencies=["scope"]),
        ]
    )
    state.model_request_count = 3
    state.main_loop_model_request_count = 3
    state.model_metrics = {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 100}
    snapshot = derive_progress(state, model_request_budget=5, token_budget=150)
    assert snapshot.current_step == "verify"
    assert snapshot.percent == 50.0
    assert snapshot.model_requests_used == 3
    assert snapshot.model_requests_remaining == 2
    assert snapshot.tokens_used == 120
    assert snapshot.tokens_remaining == 30
    assert snapshot.to_dict()["percent"] == 50.0

    pending = _state(plan=[PlanStep("scope", "Scope", progress_weight=2.0), PlanStep("verify", "Verify")])
    pending.plan[0].status = "completed"
    pending.completed_steps = ["scope"]
    assert derive_progress(pending).current_step == "verify"
    assert derive_progress(pending).percent == 66.7


def test_progress_rejects_invalid_optional_budgets() -> None:
    with pytest.raises(ValueError, match="token_budget"):
        derive_progress(_state(), token_budget=-1)


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("result.json", json.dumps({"ok": True})),
        ("result.yaml", "ok: true\nitems:\n  - one\n"),
        ("result.py", "def answer() -> int:\n    return 42\n"),
    ],
)
def test_artifact_verifier_accepts_valid_bounded_text_formats(path: str, content: str) -> None:
    payload = content.encode()
    result = ArtifactVerifier.verify(
        ArtifactSpec("output", path, max_bytes=1_024),
        ToolResult(True, "", data={"exists": True, "bytes": len(payload)}),
        content=payload,
    )
    assert result.passed is True
    assert result.size_bytes == len(payload)
    assert result.errors == ()


@pytest.mark.parametrize(
    ("path", "content", "message"),
    [
        ("result.json", b'{"broken":}', "invalid JSON"),
        ("result.yaml", b"key: [unterminated", "invalid YAML"),
        ("result.py", b"def broken(:\n", "invalid Python"),
    ],
)
def test_artifact_verifier_reports_syntax_failures(path: str, content: bytes, message: str) -> None:
    result = verify_artifact(
        ArtifactSpec("output", path),
        ToolResult(True, "", data={"exists": True, "bytes": len(content)}),
        content=content,
    )
    assert result.passed is False
    assert message in result.message


def test_artifact_verifier_fails_closed_without_content_or_existence_metadata(tmp_path) -> None:
    real_file = tmp_path / "secret.json"
    real_file.write_text('{"would": "pass"}', encoding="utf-8")
    result = verify_artifact(
        ArtifactSpec("output", "secret.json"),
        ToolResult(True, "read succeeded", data={}),
    )
    assert result.passed is False
    assert "existence was not proven" in result.message
    assert "requires supplied complete content" in result.message


def test_artifact_verifier_checks_nonempty_size_and_metadata_consistency() -> None:
    empty = verify_artifact(
        ArtifactSpec("output", "output.txt"),
        ToolResult(True, "", data={"exists": True, "bytes": 0}),
        content=b"",
    )
    assert empty.passed is False
    assert "smaller than" in empty.message
    assert "no non-whitespace text" in empty.message

    mismatch = verify_artifact(
        ArtifactSpec("output", "output.txt", max_bytes=10),
        ToolResult(True, "", data={"exists": True, "bytes": 9}),
        content=b"ok",
    )
    assert mismatch.passed is False
    assert "does not match" in mismatch.message


def test_artifact_verifier_validates_docx_structure_and_nonempty_text() -> None:
    payload = _docx_bytes("Verified report")
    result = verify_artifact(
        ArtifactSpec("report", "report.docx", max_bytes=100_000),
        ToolResult(True, "", data={"exists": True, "bytes": len(payload)}),
        content=payload,
    )
    assert result.passed is True
    assert result.detected_format == "docx"

    empty_payload = _docx_bytes("   ")
    empty = verify_artifact(
        ArtifactSpec("report", "report.docx", max_bytes=100_000),
        ToolResult(True, "", data={"exists": True, "bytes": len(empty_payload)}),
        content=empty_payload,
    )
    assert empty.passed is False
    assert "no non-empty text" in empty.message


def test_artifact_verifier_validates_bounded_managed_docx_receipt_without_content() -> None:
    payload = _docx_bytes("Verified managed report")
    spec = ArtifactSpec(MANAGED_DOCUMENT_ARTIFACT_ID, "reports/report.docx", max_bytes=100_000)
    verified = verify_artifact(
        spec,
        ToolResult(True, "", data={"exists": True, "size_bytes": len(payload), "content_complete": True}),
        content=payload,
    )
    metadata = artifact_verification_metadata(spec, verified, content=payload)

    assert verified.passed is True
    assert "docx_package_structure" in verified.checks_run
    assert "docx_nonempty_text" in verified.checks_run
    assert "content" not in metadata
    receipt = ToolResult(True, "parsed", data={ARTIFACT_VERIFICATION_METADATA_KEY: metadata})
    assert ArtifactVerifier.verify_receipt(spec, receipt).passed is True

    incomplete = dict(metadata)
    incomplete["content_complete"] = False
    rejected = ArtifactVerifier.verify_receipt(
        spec,
        ToolResult(True, "parsed", data={ARTIFACT_VERIFICATION_METADATA_KEY: incomplete}),
    )
    assert rejected.passed is False
    assert "complete content" in rejected.message

    missing_structure = dict(metadata)
    missing_structure["checks_run"] = [check for check in metadata["checks_run"] if check != "docx_package_structure"]
    rejected = ArtifactVerifier.verify_receipt(
        spec,
        ToolResult(True, "parsed", data={ARTIFACT_VERIFICATION_METADATA_KEY: missing_structure}),
    )
    assert rejected.passed is False
    assert "docx_package_structure" in rejected.message


def test_artifact_verifier_rejects_docx_zip_bomb_indicators() -> None:
    payload = _docx_bytes("Report", extra=b"A" * 40_000)
    result = verify_artifact(
        ArtifactSpec(
            "report",
            "report.docx",
            max_bytes=100_000,
            max_docx_uncompressed_bytes=100_000,
            max_docx_compression_ratio=10.0,
        ),
        ToolResult(True, "", data={"exists": True, "bytes": len(payload)}),
        content=payload,
    )
    assert result.passed is False
    assert "compression ratio" in result.message


def test_artifact_spec_rejects_unbounded_or_escaping_inputs() -> None:
    with pytest.raises(ValueError, match="project-relative"):
        ArtifactSpec("report", "../report.docx")
    with pytest.raises(ValueError, match="max_bytes"):
        ArtifactSpec("report", "report.docx", max_bytes=10**9)
    with pytest.raises(ValueError, match="unsupported"):
        ArtifactSpec("report", "report.docx", format="executable")
