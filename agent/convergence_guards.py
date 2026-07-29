"""Pure evidence parsing and guards for the convergence state machine."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from typing import Any


BROAD_EXPLORATION_FUNCTIONS = frozenset({"list_dir", "find_files", "search_code"})
TARGETED_EXPLORATION_FUNCTIONS = frozenset({"read_file", "tool_result_read"})
READ_ONLY_CAPABILITIES = frozenset(
    {
        "template.list_dir",
        "template.find_files",
        "template.search_code",
        "template.read_file",
        "tool_result.read",
        "project.read_context",
        "git.status",
        "git.diff",
        "git.log",
        "document.parse",
    }
)
SAFE_ARGUMENT_KEYS = (
    "path",
    "start_line",
    "end_line",
    "query",
    "glob",
    "pattern",
    "framework",
    "depth",
    "request_id",
    "offset",
    "max_chars",
)
MAX_IMPLEMENTATION_READ_LINES = 200
MAX_VALIDATION_ATTACHMENT_READ_CHARS = 12_000
MAX_PERSISTED_SEEN_TARGETS = 128
MAX_TARGET_KEY_CHARS = 512
VALIDATION_ATTACHMENT_CAPABILITIES = frozenset(
    {"template.run_tests", "lsp.diagnostics", "document.parse", "template.git_diff_staged"}
)


def bounded_turn(state: Any) -> int:
    value = getattr(state, "turn", 0)
    return max(0, min(value, 1_000_000)) if isinstance(value, int) and not isinstance(value, bool) else 0


def parse_arguments(arguments: str | dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, json.JSONDecodeError):
        return None, "arguments are not valid JSON"
    if not isinstance(parsed, dict):
        return None, "arguments must be an object"
    return parsed, ""


def implementation_read_denial(
    state: Any,
    function_name: str,
    arguments: str | dict[str, Any] | None,
    *,
    allowance_remaining: bool,
) -> str:
    if function_name != "read_file":
        return "only read_file can use the bounded implementation evidence exception"
    if not implementation_step_active(state):
        return "the implement step is not in progress"
    if not allowance_remaining:
        return "the bounded implementation evidence read allowance is exhausted"
    args, error = parse_arguments(arguments)
    if error:
        return error
    assert args is not None
    path = normalized_path(args.get("path"))
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    if not path:
        return "an exact path is required"
    if (
        isinstance(start_line, bool)
        or isinstance(end_line, bool)
        or not isinstance(start_line, int)
        or not isinstance(end_line, int)
        or start_line < 1
        or end_line < start_line
    ):
        return "explicit positive start_line/end_line values are required"
    if end_line - start_line + 1 > MAX_IMPLEMENTATION_READ_LINES:
        return f"the requested implementation evidence range exceeds {MAX_IMPLEMENTATION_READ_LINES} lines"
    if not path_was_read_successfully(state, path):
        return "the path was not read successfully before the exploration window closed"
    return ""


def validation_attachment_read_denial(
    state: Any,
    function_name: str,
    arguments: str | dict[str, Any] | None,
    *,
    allowance_remaining: bool,
) -> str:
    if function_name != "tool_result_read":
        return "only tool_result_read can use the bounded validation attachment exception"
    if not implementation_or_verification_step_active(state):
        return "the implement or verify step is not in progress"
    if not allowance_remaining:
        return "the bounded validation attachment read allowance is exhausted"
    args, error = parse_arguments(arguments)
    if error:
        return error
    assert args is not None
    request_id = str(args.get("request_id") or "").strip()
    offset = args.get("offset", 0)
    max_chars = args.get("max_chars", MAX_VALIDATION_ATTACHMENT_READ_CHARS)
    if not request_id:
        return "a validation attachment request_id is required"
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return "offset must be a non-negative integer"
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or max_chars < 1
        or max_chars > MAX_VALIDATION_ATTACHMENT_READ_CHARS
    ):
        return f"max_chars must be between 1 and {MAX_VALIDATION_ATTACHMENT_READ_CHARS}"
    if not is_validation_attachment(state, request_id):
        return "the request_id is not an attachment produced by a bounded validation tool in this Session"
    return ""


def has_validation_attachment(state: Any | None) -> bool:
    return any(validation_attachment_id(item) for item in getattr(state, "tool_calls", ()) or ())


def is_validation_attachment(state: Any, request_id: str) -> bool:
    return any(validation_attachment_id(item) == request_id for item in getattr(state, "tool_calls", ()) or ())


def validation_attachment_id(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    request = item.get("request") if isinstance(item.get("request"), dict) else {}
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    attachment = data.get("attachment") if isinstance(data.get("attachment"), dict) else {}
    capability = f"{request.get('tool', '')}.{request.get('action', '')}"
    if capability not in VALIDATION_ATTACHMENT_CAPABILITIES:
        if capability != "shell.run" or is_exploration_bypass("shell_run", request.get("args")):
            return ""
    request_id = str(request.get("request_id") or "").strip()
    attachment_id = str(attachment.get("request_id") or "").strip()
    return request_id if request_id and request_id == attachment_id else ""


def implementation_step_active(state: Any | None) -> bool:
    return _active_step_with_type(state, {"implement"})


def implementation_or_verification_step_active(state: Any | None) -> bool:
    return _active_step_with_type(
        state,
        {"implement", "synthesize", "generate", "render", "verify", "review"},
    )


def conditional_mutation_step_active(state: Any | None) -> bool:
    if state is None or not implementation_step_active(state):
        return False
    route = getattr(state, "task_route", {})
    reasons = route.get("reasons") if isinstance(route, dict) else None
    return isinstance(reasons, list) and "conditional-mutation" in reasons


def exploration_step_active(state: Any | None) -> bool:
    return _active_step_with_type(state, {"scope", "inspect"})


def _active_step_with_type(state: Any | None, accepted: set[str]) -> bool:
    if state is None:
        return False
    for step in getattr(state, "plan", ()) or ():
        status = step.get("status") if isinstance(step, dict) else getattr(step, "status", "")
        if step_semantic_type(step) in accepted and str(status) == "in_progress":
            return True
    return False


def step_semantic_type(step: Any) -> str:
    if isinstance(step, dict):
        value = step.get("step_type") or step.get("kind")
        step_id = step.get("id")
    else:
        value = getattr(step, "step_type", None) or getattr(step, "kind", None)
        step_id = getattr(step, "id", "")
    normalized = str(value or "").strip().lower()
    if normalized and normalized != "generic":
        return {"validate": "verify", "validation": "verify"}.get(normalized, normalized)
    return {
        "scope": "scope",
        "inspect-chunks": "inspect",
        "implement": "implement",
        "synthesize": "synthesize",
        "render-artifact": "render",
        "verify": "verify",
    }.get(str(step_id or "").strip().lower(), "generic")


def plan_requires_transition(state: Any | None) -> bool:
    if state is None:
        return False
    steps = list(getattr(state, "plan", ()) or ())
    if not steps:
        return False
    statuses = [str(step.get("status") if isinstance(step, dict) else getattr(step, "status", "")) for step in steps]
    return "in_progress" not in statuses and any(status == "pending" for status in statuses)


def normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").rstrip("/")


def path_was_read_successfully(state: Any, path: str) -> bool:
    for item in getattr(state, "tool_calls", ()) or ():
        if not isinstance(item, dict):
            continue
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        args = request.get("args") if isinstance(request.get("args"), dict) else {}
        if (
            str(request.get("tool") or "") == "template"
            and str(request.get("action") or "") == "read_file"
            and bool(result.get("success"))
            and normalized_path(args.get("path")) == path
        ):
            return True
    return False


def target_key(request: dict[str, Any]) -> str:
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    capability = f"{request.get('tool', '')}.{request.get('action', '')}"
    target: dict[str, Any] = {}
    for key in SAFE_ARGUMENT_KEYS:
        if key not in args:
            continue
        value = args[key]
        target[key] = value.strip().casefold() if isinstance(value, str) else value
    if not target:
        return ""
    key = capability + ":" + json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(key) <= MAX_TARGET_KEY_CHARS:
        return key
    return capability + ":sha256:" + hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()


def is_exploration_bypass(function_name: str, arguments: str | dict[str, Any] | None) -> bool:
    if function_name == "shell_run":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return True
        command = str((parsed or {}).get("command") or "") if isinstance(parsed, dict) else ""
        return not is_bounded_validation_command(command)
    if function_name == "python_run":
        try:
            json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return True
        return True
    return False


def is_bounded_validation_command(command: str) -> bool:
    value = str(command or "").strip()
    if not value or any(marker in value for marker in ("\n", ";", "|", "&", "<", ">", "$", "`")):
        return False
    try:
        args = shlex.split(value, posix=True)
    except ValueError:
        return False
    if not args or len(args) > 32:
        return False
    program = args[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    rest = [item.casefold() for item in args[1:]]
    mutation_flags = {
        "--fix",
        "--write",
        "--apply",
        "--update",
        "--bless",
        "--accept",
        "--install-types",
    }
    if any(
        item in mutation_flags
        or any(item.startswith(flag + "=") for flag in mutation_flags)
        or (item.startswith("-w") and not item.startswith("--"))
        for item in rest
    ):
        return False
    package_scripts = {"test", "typecheck", "check", "lint"}
    if program in {"npm", "pnpm", "yarn", "bun"}:
        if rest and rest[0] == "test":
            return True
        return len(rest) >= 2 and rest[0] == "run" and rest[1] in package_scripts
    if program in {"pytest", "py.test"}:
        return True
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", program):
        return len(rest) >= 2 and rest[:2] == ["-m", "pytest"]
    if program == "ruff":
        return bool(rest) and (rest[0] == "check" or rest[:2] == ["format", "--check"])
    if program in {"pyright", "mypy"}:
        return True
    if program == "tsc":
        return "--noemit" in rest
    if program == "npx":
        return len(rest) >= 2 and rest[0] == "tsc" and "--noemit" in rest[1:]
    if program == "cargo":
        return bool(rest) and rest[0] in {"test", "check", "clippy"}
    if program == "go":
        return bool(rest) and rest[0] == "test"
    if program in {"mvn", "mvnw", "gradle", "gradlew"}:
        return any(item in {"test", "check", "verify"} for item in rest)
    return program == "git" and rest == ["diff", "--check"]


__all__ = [
    "BROAD_EXPLORATION_FUNCTIONS",
    "MAX_IMPLEMENTATION_READ_LINES",
    "MAX_PERSISTED_SEEN_TARGETS",
    "MAX_TARGET_KEY_CHARS",
    "MAX_VALIDATION_ATTACHMENT_READ_CHARS",
    "READ_ONLY_CAPABILITIES",
    "TARGETED_EXPLORATION_FUNCTIONS",
    "bounded_turn",
    "conditional_mutation_step_active",
    "exploration_step_active",
    "has_validation_attachment",
    "implementation_or_verification_step_active",
    "implementation_read_denial",
    "implementation_step_active",
    "is_bounded_validation_command",
    "is_exploration_bypass",
    "is_validation_attachment",
    "normalized_path",
    "path_was_read_successfully",
    "plan_requires_transition",
    "step_semantic_type",
    "target_key",
    "validation_attachment_id",
    "validation_attachment_read_denial",
]
