from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .base import ToolRequest
from .registry import ToolCapability


DEFAULT_DENY_SHELL_PATTERNS = (
    r"(?i)(?:^|[;&|]\s*)sudo(?:\s|$)",
    r"(?i)(?:^|[;&|]\s*)su(?:\s|$)",
    r"(?i)\b(?:shutdown|reboot|poweroff|halt|mkfs(?:\.[a-z0-9]+)?|fdisk|parted)\b",
    r"(?i)\brm\s+(?:-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\s+(?:--no-preserve-root\s+)?(?:/|~|\$HOME|\.\.?)(?:\s|$)",
    r"(?i)\bchmod\s+(?:-[a-z]*R[a-z]*\s+)?777\s+/(?:\s|$)",
)


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""


class PermissionManager:
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve()

    def evaluate(
        self,
        request: ToolRequest,
        capability: ToolCapability,
        *,
        super_yolo: bool = False,
    ) -> PermissionDecision:
        if not capability.enabled:
            return PermissionDecision(False, f"tool capability is disabled: {capability.name}")
        if not capability.available:
            reason = capability.unavailable_reason or "required dependency is not available"
            return PermissionDecision(False, f"tool capability is unavailable: {capability.name}: {reason}")
        if super_yolo:
            return PermissionDecision(True, "SUPER YOLO bypassed permission policy")
        if not bool(self.config.get("permissions.enforce", True)):
            return PermissionDecision(True)
        denied = self.config.get("permissions.deny_capabilities", [])
        if isinstance(denied, list) and capability.name in {str(item) for item in denied}:
            return PermissionDecision(False, f"capability denied by policy: {capability.name}")

        cwd_decision = self._check_working_directory(request.args.get("cwd"))
        if not cwd_decision.allowed:
            return cwd_decision
        timeout = request.args.get("timeout")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
                return PermissionDecision(False, "tool timeout must be a positive integer")
            if timeout > capability.timeout_seconds:
                return PermissionDecision(
                    False,
                    f"requested timeout exceeds capability limit: {timeout}s > {capability.timeout_seconds}s",
                )
        if capability.name == "shell.run":
            return self._check_shell(str(request.args.get("command") or ""))
        if capability.name == "docker.run":
            return self._check_docker(request.args.get("args"))
        return PermissionDecision(True)

    def _check_working_directory(self, cwd: Any) -> PermissionDecision:
        if not cwd or not bool(self.config.get("permissions.restrict_cwd_to_project", True)):
            return PermissionDecision(True)
        candidate = Path(str(cwd)).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        try:
            candidate.resolve().relative_to(self.project_root)
        except ValueError:
            return PermissionDecision(False, f"working directory is outside the current project: {candidate}")
        return PermissionDecision(True)

    def _check_shell(self, command: str) -> PermissionDecision:
        if not command.strip():
            return PermissionDecision(False, "shell command is empty")
        if "\x00" in command:
            return PermissionDecision(False, "shell command contains a NUL byte")
        configured = self.config.get("permissions.deny_shell_patterns", list(DEFAULT_DENY_SHELL_PATTERNS))
        patterns = configured if isinstance(configured, list) else list(DEFAULT_DENY_SHELL_PATTERNS)
        for pattern in patterns:
            try:
                if re.search(str(pattern), command):
                    return PermissionDecision(False, f"shell command denied by policy pattern: {pattern}")
            except re.error:
                continue
        return PermissionDecision(True)

    @staticmethod
    def _check_docker(args: Any) -> PermissionDecision:
        values = [str(item) for item in args] if isinstance(args, list) else []
        normalized = " ".join(values)
        if any(value == "--privileged" or value.startswith("--privileged=") for value in values):
            return PermissionDecision(False, "docker --privileged is denied by default")
        if re.search(r"(?:^|\s)(?:-v|--volume)(?:=|\s+)/(?:[:]|$)", normalized):
            return PermissionDecision(False, "mounting the host root into Docker is denied by default")
        if re.search(r"(?:^|\s)--mount(?:=|\s+)[^\s]*(?:src|source)=/(?:,|$)", normalized, re.IGNORECASE):
            return PermissionDecision(False, "mounting the host root into Docker is denied by default")
        if any(
            value in {"--pid=host", "--network=host", "--ipc=host", "--uts=host"}
            or value.startswith("--device=")
            or value == "--device"
            for value in values
        ):
            return PermissionDecision(False, "Docker host namespace or device access is denied by default")
        if any("/var/run/docker.sock" in value for value in values):
            return PermissionDecision(False, "mounting the Docker socket is denied by default")
        return PermissionDecision(True)
