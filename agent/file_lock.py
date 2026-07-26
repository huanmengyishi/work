from __future__ import annotations

import errno
import os
from typing import Any


class FileLockUnavailable(RuntimeError):
    pass


def lock_exclusive(handle: Any, *, nonblocking: bool = False) -> None:
    """Acquire a one-process file lock on POSIX or native Windows."""

    if os.name == "nt":
        _windows_lock(handle, nonblocking=nonblocking)
        return
    import fcntl

    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        fcntl.flock(handle.fileno(), flags)
    except OSError as exc:
        if nonblocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise FileLockUnavailable("file lock is already held") from exc
        raise


def unlock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            return
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _windows_lock(handle: Any, *, nonblocking: bool) -> None:
    import msvcrt

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write("\0")
        handle.flush()
    handle.seek(0)
    mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
    try:
        msvcrt.locking(handle.fileno(), mode, 1)
    except OSError as exc:
        raise FileLockUnavailable("file lock is already held") from exc


__all__ = ["FileLockUnavailable", "lock_exclusive", "unlock"]
