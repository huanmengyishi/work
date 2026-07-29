from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from .artifact import (
    MANAGED_DOCUMENT_ARTIFACT_ID,
    MAX_ARTIFACT_BYTES_HARD_LIMIT,
    ArtifactSpec,
    ArtifactVerifier,
)
from .artifact_registry import ArtifactRegistry
from .constants import (
    SINGLE_VALIDATION_CAPABILITIES,
    SINGLE_VALIDATION_MODEL_FUNCTIONS,
    VALIDATION_SHELL_PROGRAMS,
)
from .state import AgentState
from .task_router import TaskRoute
from .runtime_support import (
    _DATE_LITERAL_RE,
    _date_key,
    _date_keys_from_text,
)


class RuntimeValidationMixin:
    @staticmethod
    def _failure_count(state: AgentState) -> int:
        failed_tools = sum(
            1
            for call in state.tool_calls[-20:]
            if isinstance(call, dict)
            and isinstance(call.get("result"), dict)
            and not bool(call["result"].get("success", False))
        )
        return min(10, max(state.failure_count, failed_tools + int(bool(state.error))))

    @staticmethod
    def _single_validation_requested(state: AgentState) -> bool:
        reasons = (state.task_route or {}).get("reasons")
        return isinstance(reasons, list) and "single-validation" in reasons

    @classmethod
    def _single_validation_used(cls, state: AgentState) -> bool:
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if data.get("runtime_denied") is True or data.get("not_executed") is True:
                continue
            capability = f"{request.get('tool', '')}.{request.get('action', '')}"
            if capability in SINGLE_VALIDATION_CAPABILITIES:
                return True
            if capability == "shell.run" and cls._looks_like_validation_shell(request.get("args")):
                return True
        return False

    @classmethod
    def _is_validation_model_call(cls, function_name: str, arguments: str | dict[str, Any] | None) -> bool:
        return cls._validation_model_call_count(function_name, arguments) > 0

    @classmethod
    def _validation_model_call_count(cls, function_name: str, arguments: str | dict[str, Any] | None) -> int:
        if function_name in SINGLE_VALIDATION_MODEL_FUNCTIONS:
            return 1
        if function_name != "shell_run":
            return 0
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return 0
        if not isinstance(parsed, dict):
            return 0
        return cls._shell_command_validation_count(str(parsed.get("command") or ""))

    @classmethod
    def _looks_like_validation_shell(cls, arguments: str | dict[str, Any] | None) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(parsed, dict):
            return False
        command = str(parsed.get("command") or "").strip()
        if not command:
            return False
        return cls._shell_command_contains_validation(command)

    @classmethod
    def _shell_command_contains_validation(cls, command: str) -> bool:
        return cls._shell_command_validation_count(command) > 0

    @classmethod
    def _shell_command_validation_count(cls, command: str) -> int:
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            return False
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token and set(token) <= {";", "&", "|"}:
                if segments[-1]:
                    segments.append([])
                continue
            segments[-1].append(token)
        return sum(cls._validation_argv_count(segment) for segment in segments if segment)

    @classmethod
    def _argv_is_validation(cls, args: list[str]) -> bool:
        return cls._validation_argv_count(args) > 0

    @classmethod
    def _validation_argv_count(cls, args: list[str]) -> int:
        if not args:
            return 0
        program = args[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
        rest = [item.casefold() for item in args[1:]]
        if program in {"bash", "dash", "sh", "zsh"}:
            command_index = next(
                (
                    index
                    for index, flag in enumerate(rest)
                    if flag == "-c" or (flag.startswith("-") and "c" in flag[1:])
                ),
                None,
            )
            if command_index is None or command_index + 2 > len(args) - 1:
                return 0
            return cls._shell_command_validation_count(args[command_index + 2])
        if program == "env":
            index = 1
            while index < len(args):
                item = args[index]
                option = item.casefold()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", item):
                    index += 1
                    continue
                if option == "--":
                    index += 1
                    break
                if option in {"-i", "--ignore-environment", "-0", "--null"}:
                    index += 1
                    continue
                if option in {"-u", "--unset", "-c", "--chdir"}:
                    index += 2
                    continue
                if option.startswith(("--unset=", "--chdir=")):
                    index += 1
                    continue
                if option.startswith("-"):
                    return 0
                break
            return cls._validation_argv_count(args[index:])
        if program in {"command", "exec"}:
            index = 1
            while index < len(args) and args[index].startswith("-"):
                option = args[index].casefold()
                if option == "--":
                    index += 1
                    break
                if program == "command" and option == "-v":
                    return 0
                if program == "exec" and option == "-a":
                    index += 2
                else:
                    index += 1
            return cls._validation_argv_count(args[index:])
        if program == "timeout":
            index = 1
            while index < len(args) and args[index].startswith("-"):
                option = args[index].casefold()
                index += 2 if option in {"-k", "--kill-after", "-s", "--signal"} else 1
            if index >= len(args):
                return 0
            return cls._validation_argv_count(args[index + 1 :])
        if program == "uv":
            return cls._validation_argv_count(args[2:]) if rest and rest[0] == "run" else 0
        if program in {"tox", "nox"}:
            return 1
        if program == "make":
            return int(any(item in {"test", "tests", "check", "lint", "typecheck", "verify"} for item in rest))
        if program in {"npm", "pnpm", "yarn", "bun"}:
            command = cls._package_manager_command(rest)
            if not command:
                return 0
            if command[0] in {"test", "check", "lint", "typecheck"}:
                return 1
            return int(
                len(command) >= 2
                and command[0] in {"run", "run-script"}
                and command[1] in {"test", "check", "lint", "typecheck", "build"}
            )
        if re.fullmatch(r"python(?:3(?:\.\d+)?)?", program):
            index = 0
            while index < len(rest):
                option = rest[index]
                if option == "-m":
                    return int(index + 1 < len(rest) and rest[index + 1] in {"pytest", "mypy", "pyright"})
                if option in {"-c", "--check-hash-based-pycs"} or not option.startswith("-"):
                    return 0
                index += 2 if option in {"-w", "-x"} and index + 1 < len(rest) else 1
            return 0
        if program == "npx":
            return int(bool(rest) and rest[0] in {"tsc", "eslint", "jest", "vitest", "mocha"})
        if program == "cargo":
            return int(bool(rest) and rest[0] in {"test", "check", "clippy", "build"})
        if program == "go":
            return int(bool(rest) and rest[0] in {"test", "vet"})
        if program in {"mvn", "mvnw", "gradle", "gradlew"}:
            return int(any(item in {"test", "check", "verify"} for item in rest))
        if program == "git":
            return int(rest[:2] == ["diff", "--check"])
        return int(program in VALIDATION_SHELL_PROGRAMS)

    @staticmethod
    def _package_manager_command(rest: list[str]) -> list[str]:
        value_options = {"--prefix", "--workspace", "--cwd", "--dir", "-c"}
        index = 0
        while index < len(rest) and rest[index].startswith("-"):
            option = rest[index]
            if option in value_options and index + 1 < len(rest):
                index += 2
            else:
                index += 1
        return rest[index:]

    @classmethod
    def _completion_issue(cls, state: AgentState, final: str) -> str:
        issues = list(cls._execution_evidence_issues(state))
        if not final.strip():
            issues.append("the model returned an empty final answer")
        else:
            lowered = final.strip().lower()
            progress_only = (
                "need to use" in lowered
                or "let me " in lowered
                or "i will " in lowered
                or "需要用" in final
                or "接下来" in final
                or "让我" in final
            )
            if progress_only and len(final) < 500:
                issues.append("the response is only a progress note and explicitly describes remaining work")
        return "; ".join(dict.fromkeys(issues))

    @classmethod
    def _execution_evidence_issue(cls, state: AgentState) -> str:
        """Return missing execution evidence independently of answer prose.

        The same gate controls ordinary completion and whether the soft tool
        target may close tool execution.  This prevents a model-authored plan
        status from sending an artifact or validation task into tool-free final
        synthesis before the corresponding tool evidence exists.
        """

        return "; ".join(cls._execution_evidence_issues(state))

    @classmethod
    def _execution_evidence_issues(cls, state: AgentState) -> tuple[str, ...]:
        """Return all independently actionable execution-evidence gaps."""

        issues: list[str] = []
        requires_plan = bool((state.task_route or {}).get("require_plan")) or bool(
            (state.task_strategy or {}).get("require_plan")
        )
        if requires_plan:
            if not state.plan:
                issues.append("the selected execution mode requires a Task Graph, but no plan exists")
            else:
                incomplete = [step.id for step in state.plan if not state.plan_step_satisfied(step)]
                if incomplete:
                    shown = ", ".join(incomplete[:5])
                    suffix = "..." if len(incomplete) > 5 else ""
                    issues.append(f"required Task Graph steps are still incomplete: {shown}{suffix}")
                else:
                    plan_evidence_issue = cls._plan_evidence_issue(state)
                    if plan_evidence_issue:
                        issues.append(plan_evidence_issue)
        reasons = set((state.task_route or {}).get("reasons") or [])
        if "single-validation" in reasons:
            executed = cls._executed_non_plan_calls(state)
            if not any(cls._recorded_call_is_validation(item) for item in executed):
                issues.append("the single-validation task has no executed validation attempt")
        artifact_issue = cls._artifact_evidence_issue(state, reasons)
        if artifact_issue:
            issues.append(artifact_issue)
        return tuple(dict.fromkeys(issues))

    @classmethod
    def _artifact_evidence_issue(cls, state: AgentState, reasons: set[str]) -> str:
        if "artifact-required" not in reasons:
            return ""

        registry_handled, registry_issue = ArtifactRegistry.completion_issue(state, reasons)
        if registry_handled:
            # The bounded registry is updated before hot-window pruning and is
            # authoritative both before and after old ToolResults leave State.
            return registry_issue

        route = TaskRoute.from_dict(state.task_route or {})
        successful = cls._successful_tool_records(state)
        directory_writes = [
            item
            for item in successful
            if str((item.get("request") or {}).get("tool") or "") == "template"
            and str((item.get("request") or {}).get("action") or "") == "make_dir"
        ]
        if "directory-artifact-required" in reasons and not directory_writes:
            return "the requested output directory has no successful managed make_dir evidence"
        unmatched_directories = [
            hint
            for hint in route.directory_hints
            if not any(
                cls._same_recorded_path(
                    state,
                    hint,
                    str(((item.get("result") or {}).get("data") or {}).get("path") or ""),
                )
                for item in directory_writes
            )
        ]
        if unmatched_directories:
            return "the requested output directory has no successful managed make_dir evidence matching: " + ", ".join(
                unmatched_directories[:8]
            )
        directory_hints = tuple(hint for hint in route.artifact_hints if "." not in hint)
        if directory_hints:
            unmatched_directories = [
                hint
                for hint in directory_hints
                if not any(
                    cls._artifact_hint_matches_path(
                        hint,
                        str(((item.get("result") or {}).get("data") or {}).get("path") or "")
                        or str(((item.get("request") or {}).get("args") or {}).get("path") or ""),
                    )
                    for item in directory_writes
                )
            ]
            if unmatched_directories:
                return (
                    "the requested output directory has no successful managed make_dir evidence matching: "
                    + ", ".join(unmatched_directories[:8])
                )

        active_applies = [
            item
            for item in cls._active_file_applies(state)
            if item["after_exists"] is True or (item["after_exists"] is None and route.schema_version < 2)
        ]
        file_hints = tuple(hint for hint in route.artifact_hints if hint not in directory_hints)
        needs_file_artifact = (
            "directory-artifact-required" not in reasons or bool(file_hints) or "word-artifact-required" in reasons
        )
        if not needs_file_artifact:
            return ""
        if not active_applies:
            return "the requested output artifact has no active successful managed-write evidence"

        unmatched_hints = [
            hint
            for hint in file_hints
            if not any(cls._artifact_hint_matches_path(hint, str(item["path"])) for item in active_applies)
        ]
        if unmatched_hints:
            return (
                "the requested output artifact has no active successful managed-write evidence matching: "
                + ", ".join(unmatched_hints[:8])
            )
        if "word-artifact-required" not in reasons:
            return ""

        word_hints = tuple(hint for hint in route.artifact_hints if hint.lower().endswith(".docx"))
        word_applies: list[dict[str, Any]] = []
        if word_hints:
            for hint in word_hints:
                matching = [
                    item
                    for item in active_applies
                    if cls._artifact_hint_matches_path(hint, str(item["path"]))
                    and str(item["path"]).lower().endswith(".docx")
                ]
                if matching:
                    word_applies.append(max(matching, key=lambda item: (item["round"], item["index"])))
        else:
            matching = [item for item in active_applies if str(item["path"]).lower().endswith(".docx")]
            if matching:
                word_applies.append(max(matching, key=lambda item: (item["round"], item["index"])))
        if not word_applies:
            return "the requested Word artifact has no active applied .docx preview"

        seen_applies: set[tuple[str, str, str]] = set()
        for applied in word_applies:
            identity = (str(applied["path"]), str(applied["preview_id"]), str(applied["snapshot_id"]))
            if identity in seen_applies:
                continue
            seen_applies.add(identity)
            issue = cls._word_artifact_evidence_issue(
                state,
                applied=applied,
                successful=successful,
                route_schema=route.schema_version,
            )
            if issue:
                return issue
        return ""

    @classmethod
    def _word_artifact_evidence_issue(
        cls,
        state: AgentState,
        *,
        applied: dict[str, Any],
        successful: list[dict[str, Any]],
        route_schema: int,
    ) -> str:
        artifact_path = str(applied["path"])
        artifact_index = int(applied["index"])
        artifact_preview_id = str(applied["preview_id"] or "")
        previews: list[dict[str, Any]] = []
        for index, item in enumerate(state.tool_calls):
            if item not in successful:
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if (str(request.get("tool") or ""), str(request.get("action") or "")) != ("document", "render_docx"):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            previews.append(
                {
                    "record": item,
                    "round": int(item.get("round") or 0),
                    "index": index,
                    "path": str(data.get("path") or ""),
                    "preview_id": str(data.get("preview_id") or ""),
                }
            )
        path_previews = [item for item in previews if cls._same_recorded_path(state, str(item["path"]), artifact_path)]
        if not path_previews:
            return "the requested Word artifact has no matching document_render_docx preview"
        if route_schema >= 2 and not artifact_preview_id:
            return "the requested Word artifact apply record is missing preview lineage"
        if artifact_preview_id:
            matching_previews = [item for item in path_previews if item["preview_id"] == artifact_preview_id]
        else:
            matching_previews = path_previews
        if not matching_previews:
            return "the requested Word artifact apply does not match its document_render_docx preview_id"
        latest_path_preview = max(path_previews, key=lambda item: (item["round"], item["index"]))
        if (
            latest_path_preview["preview_id"]
            and artifact_preview_id
            and latest_path_preview["preview_id"] != artifact_preview_id
        ) or (latest_path_preview["round"], latest_path_preview["index"]) > (
            int(applied["round"]),
            artifact_index,
        ):
            return "the latest generated document preview has not been applied"

        parse_results: list[dict[str, Any]] = []
        for index, item in enumerate(state.tool_calls):
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            request_args = request.get("args") if isinstance(request.get("args"), dict) else {}
            if (
                index > artifact_index
                and (str(request.get("tool") or ""), str(request.get("action") or "")) == ("document", "parse")
                and bool(result.get("success"))
                and cls._same_recorded_path(
                    state,
                    str(request_args.get("path") or ""),
                    artifact_path,
                )
            ):
                parse_results.append(result)
        if not parse_results:
            return "the generated document has not been re-opened with document_parse"
        try:
            artifact_spec = ArtifactSpec(
                MANAGED_DOCUMENT_ARTIFACT_ID,
                artifact_path,
                format="docx",
                max_bytes=MAX_ARTIFACT_BYTES_HARD_LIMIT,
            )
        except ValueError:
            return "the applied Word artifact path cannot be validated as managed project-relative evidence"
        receipt_results = [ArtifactVerifier.verify_receipt(artifact_spec, result) for result in parse_results]
        if not any(result.passed for result in receipt_results):
            evidence_errors = next((result.errors for result in reversed(receipt_results) if result.errors), ())
            detail = f": {evidence_errors[0]}" if evidence_errors else ""
            return (
                "the re-opened Word artifact has no managed metadata proving complete, non-empty, "
                f"structurally valid DOCX content{detail}"
            )

        latest_render_record = max(matching_previews, key=lambda item: (item["round"], item["index"]))["record"]
        render_result = (
            latest_render_record.get("result") if isinstance(latest_render_record.get("result"), dict) else {}
        )
        render_data = render_result.get("data") if isinstance(render_result.get("data"), dict) else {}
        generated_date_values = {
            str(item) for item in render_data.get("generated_metadata_dates", []) if str(item).strip()
        }
        if not generated_date_values:
            render_request = (
                latest_render_record.get("request") if isinstance(latest_render_record.get("request"), dict) else {}
            )
            render_args = render_request.get("args") if isinstance(render_request.get("args"), dict) else {}
            markdown = str(render_args.get("markdown") or "")
            generated_date_values = {
                date
                for line in markdown.splitlines()
                if ("生成" in line or "汇总" in line or "报告" in line) and ("时间" in line or "日期" in line)
                for date in _DATE_LITERAL_RE.findall(line)
            }
        generated_dates = {key for value in generated_date_values if (key := _date_key(value)) is not None}
        invalid_generated_dates = {value for value in generated_date_values if _date_key(value) is None}
        allowed_dates = _date_keys_from_text(state.objective + "\n" + state.user_request)
        allowed_sources = {
            ("document", "parse"),
            ("ocr", "parse"),
            ("template", "read_file"),
        }
        for index, item in enumerate(state.tool_calls):
            if index >= artifact_index:
                break
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if not bool(result.get("success")):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            source = (str(request.get("tool") or ""), str(request.get("action") or ""))
            if source not in allowed_sources:
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            allowed_dates.update(_date_keys_from_text(str(data.get("date_literals") or "")))
            allowed_dates.update(_date_keys_from_text(str(result.get("stdout") or "")))
        unsupported_date_keys = generated_dates - allowed_dates
        unsupported_dates = sorted(
            invalid_generated_dates
            | {
                value
                for value in generated_date_values
                if (key := _date_key(value)) is not None and key in unsupported_date_keys
            }
        )
        if unsupported_dates:
            return (
                "the generated document contains unsupported generation-date metadata not present in the "
                "request or source documents: " + ", ".join(unsupported_dates[:5])
            )
        return ""

    @staticmethod
    def _artifact_hint_matches_path(hint: str, path: str) -> bool:
        normalized_hint = str(hint).strip()
        normalized_path = str(path).strip().replace("\\", "/").rstrip("/")
        if not normalized_hint or not normalized_path:
            return False
        basename = normalized_path.rsplit("/", maxsplit=1)[-1]
        if normalized_hint.startswith("."):
            return basename.lower().endswith(normalized_hint.lower())
        return basename == normalized_hint

    @staticmethod
    def _successful_tool_records(state: AgentState) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if bool(result.get("success")) and not bool(data.get("not_executed")):
                records.append(item)
        return records

    @classmethod
    def _active_file_applies(cls, state: AgentState) -> list[dict[str, Any]]:
        successful = cls._successful_tool_records(state)
        undone_snapshots = {
            str(((item.get("result") or {}).get("data") or {}).get("snapshot_id") or "")
            for item in successful
            if str((item.get("request") or {}).get("tool") or "") == "file"
            and str((item.get("request") or {}).get("action") or "") == "undo"
        }
        undone_snapshots.discard("")
        applies: list[dict[str, Any]] = []
        for index, item in enumerate(state.tool_calls):
            if item not in successful:
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if (str(request.get("tool") or ""), str(request.get("action") or "")) != ("file", "apply"):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            request_args = request.get("args") if isinstance(request.get("args"), dict) else {}
            snapshot_id = str(data.get("snapshot_id") or "")
            if snapshot_id and snapshot_id in undone_snapshots:
                continue
            applies.append(
                {
                    "record": item,
                    "round": int(item.get("round") or 0),
                    "index": index,
                    "path": str(data.get("path") or ""),
                    "preview_id": str(data.get("preview_id") or request_args.get("preview_id") or ""),
                    "snapshot_id": snapshot_id,
                    "before_exists": data.get("before_exists") if isinstance(data.get("before_exists"), bool) else None,
                    "after_exists": data.get("after_exists") if isinstance(data.get("after_exists"), bool) else None,
                }
            )
        latest_by_path: dict[Path, dict[str, Any]] = {}
        for item in applies:
            path_key = cls._normalized_recorded_path(state, str(item["path"]))
            if path_key is not None:
                latest_by_path[path_key] = item
        return sorted(latest_by_path.values(), key=lambda item: int(item["index"]))

    @classmethod
    def _plan_evidence_issue(cls, state: AgentState) -> str:
        executed = cls._executed_non_plan_calls(state)
        if not executed:
            return "the completed Task Graph has no executed non-plan tool evidence"
        required_rules = {
            rule
            for step in state.plan
            if state.plan_step_satisfied(step) and step.step_type in {"verify", "review"}
            for rule in step.validation_rules
        }
        if "managed_validation" in required_rules and not any(
            cls._recorded_call_is_validation(item) for item in executed
        ):
            return "the completed verification step has no executed managed validation attempt"
        if "document_parse" in required_rules and not any(
            str((item.get("request") or {}).get("tool") or "") == "document"
            and str((item.get("request") or {}).get("action") or "") == "parse"
            and bool((item.get("result") or {}).get("success"))
            for item in executed
        ):
            return "the completed document verification step has no successful document_parse evidence"
        return ""

    @staticmethod
    def _executed_non_plan_calls(state: AgentState) -> list[dict[str, Any]]:
        executed: list[dict[str, Any]] = []
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if not request or str(request.get("tool") or "") == "agent":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if bool(data.get("runtime_denied")) or bool(data.get("not_executed")):
                continue
            executed.append(item)
        return executed

    @classmethod
    def _recorded_call_is_validation(cls, item: dict[str, Any]) -> bool:
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        capability = f"{request.get('tool', '')}.{request.get('action', '')}"
        if capability in SINGLE_VALIDATION_CAPABILITIES:
            return True
        return capability == "shell.run" and cls._looks_like_validation_shell(request.get("args"))

    @staticmethod
    def _normalized_recorded_path(state: AgentState, value: str) -> Path | None:
        raw = str(value).strip().replace("\\", "/")
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            root = Path(str((state.project or {}).get("root") or state.working_directory))
            path = root / path
        return path.resolve(strict=False)

    @classmethod
    def _same_recorded_path(cls, state: AgentState, left: str, right: str) -> bool:
        left_value = cls._normalized_recorded_path(state, left)
        right_value = cls._normalized_recorded_path(state, right)
        if left_value is None or right_value is None:
            return False
        return left_value == right_value


__all__ = ["RuntimeValidationMixin"]
