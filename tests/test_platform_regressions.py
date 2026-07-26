from __future__ import annotations

import errno
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tomllib

import pytest

from agent import file_lock
from agent.capability_health import CapabilityHealthManager
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.tools import ToolManager
from agent.tools.base import run_command


def test_cli_import_does_not_require_posix_fcntl() -> None:
    script = """
import builtins

original_import = builtins.__import__

def import_without_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_fcntl
import agent.cli
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr


def test_windows_file_lock_initializes_and_locks_one_byte(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, int, int]] = []
    fake_msvcrt = SimpleNamespace(
        LK_NBLCK=11,
        LK_LOCK=12,
        LK_UNLCK=13,
        locking=lambda fd, mode, count: calls.append((fd, mode, count)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(file_lock, "os", SimpleNamespace(name="nt", SEEK_END=os.SEEK_END))
    path = tmp_path / "windows.lock"

    with path.open("a+") as handle:
        file_lock.lock_exclusive(handle, nonblocking=True)
        file_lock.unlock(handle)
        descriptor = handle.fileno()

    assert path.read_bytes() == b"\0"
    assert calls == [(descriptor, fake_msvcrt.LK_NBLCK, 1), (descriptor, fake_msvcrt.LK_UNLCK, 1)]


def test_windows_nonblocking_lock_error_has_portable_exception(tmp_path: Path, monkeypatch) -> None:
    def unavailable(_fd: int, _mode: int, _count: int) -> None:
        raise OSError("locked")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=11, LK_LOCK=12, LK_UNLCK=13, locking=unavailable)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(file_lock, "os", SimpleNamespace(name="nt", SEEK_END=os.SEEK_END))

    with (tmp_path / "windows.lock").open("a+") as handle:
        with pytest.raises(file_lock.FileLockUnavailable, match="already held"):
            file_lock.lock_exclusive(handle, nonblocking=True)


def test_posix_nonblocking_lock_maps_eacces_to_portable_exception(tmp_path: Path, monkeypatch) -> None:
    def unavailable(_fd: int, _flags: int) -> None:
        raise PermissionError(errno.EACCES, "locked")

    fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=unavailable)
    monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl)
    monkeypatch.setattr(file_lock, "os", SimpleNamespace(name="posix"))

    with (tmp_path / "posix.lock").open("a+") as handle:
        with pytest.raises(file_lock.FileLockUnavailable, match="already held"):
            file_lock.lock_exclusive(handle, nonblocking=True)


@pytest.mark.skipif(os.name == "nt", reason="shell=True uses the platform shell; this checks POSIX quoting")
def test_shell_mode_preserves_metacharacters_inside_one_argument(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    payload = f"literal; touch {marker}"

    result = run_command(["printf", "%s", payload], cwd=tmp_path, timeout=5, shell=True)

    assert result.success is True
    assert result.stdout == payload
    assert not marker.exists()


def test_tool_manager_uses_injected_capability_health(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    injected = CapabilityHealthManager(config, "injected-health")

    manager = ToolManager(config, project, memory, health=injected)

    assert manager.health is injected


def test_optional_dependency_minimums_match_requirements_file() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = {
        line.split(">=", maxsplit=1)[0].lower(): line
        for raw_line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }

    assert requirements["chromadb"] == project["optional-dependencies"]["vector"][0]
    assert requirements["playwright"] == project["optional-dependencies"]["browser"][0]


def test_ruff_version_and_lint_contract_are_release_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "ruff==0.15.21" in configuration["project"]["optional-dependencies"]["dev"]
    assert configuration["tool"]["ruff"]["required-version"] == "==0.15.21"
    assert configuration["tool"]["ruff"]["lint"]["select"] == ["E4", "E7", "E9", "F"]
