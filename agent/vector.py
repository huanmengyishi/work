from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class VectorStatus:
    enabled: bool
    reason: str


class OptionalChromaStore:
    """Optional Chroma adapter.

    The core runtime intentionally works without Chroma. This keeps the CLI usable
    on fresh WSL installs and lets vector search be enabled after dependencies are
    installed.
    """

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self._configured_enabled = bool(enabled)
        self._load_attempted = False
        self._load_lock = Lock()
        self._client = None
        self._collection = None
        self.status = (
            VectorStatus(True, "enabled by config; loads on first vector operation")
            if self._configured_enabled
            else VectorStatus(False, "disabled by config")
        )

    def _load(self) -> VectorStatus:
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            return VectorStatus(False, f"chromadb not available: {exc}")
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection("memories")
        except Exception as exc:
            return VectorStatus(False, f"chromadb init failed: {exc}")
        return VectorStatus(True, "enabled")

    def _ensure_loaded(self) -> bool:
        if not self._configured_enabled:
            return False
        if self._collection is not None:
            return True
        with self._load_lock:
            if self._collection is not None:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True
            self.status = self._load()
            return self.status.enabled

    def reset_load_failure(self) -> bool:
        """Allow one later lazy load after an explicitly acknowledged failure.

        Failed imports or initialization remain sticky by default so ordinary
        Memory operations cannot repeatedly probe an unavailable heavy
        dependency. Callers may invoke this after installing or repairing
        ChromaDB; the next real vector operation then makes one new load
        attempt. Concurrent reset/load calls share the same lock.
        """

        if not self._configured_enabled:
            return False
        with self._load_lock:
            if not self._load_attempted or self._collection is not None or self.status.enabled:
                return False
            self._client = None
            self._collection = None
            self._load_attempted = False
            self.status = VectorStatus(True, "load failure reset; retry on next vector operation")
            return True

    def is_enabled(self) -> bool:
        return self._configured_enabled and (not self._load_attempted or self.status.enabled)

    def upsert_memory(
        self,
        *,
        memory_id: int,
        project_id: str | None,
        kind: str,
        title: str,
        content: str,
        tags: list[str],
    ) -> bool:
        if not self._ensure_loaded() or self._collection is None:
            return False
        document = f"{title}\n\n{content}".strip()
        metadata: dict[str, Any] = {
            "memory_id": memory_id,
            "project_id": project_id or "__global__",
            "kind": kind,
            "title": title,
            "tags": ",".join(tags),
        }
        try:
            self._collection.upsert(
                ids=[f"memory:{memory_id}"],
                documents=[document],
                metadatas=[metadata],
            )
            return True
        except Exception as exc:
            self.status = VectorStatus(False, f"chromadb upsert failed: {exc}")
            return False

    def query_memory_ids(self, *, query: str, project_id: str | None, limit: int) -> list[int]:
        if not query.strip():
            return []
        if not self._ensure_loaded() or self._collection is None:
            return []
        try:
            count = self._collection.count()
            if count <= 0:
                return []
            result = self._collection.query(
                query_texts=[query],
                n_results=min(max(limit * 4, limit), count),
                include=["metadatas"],
            )
        except Exception as exc:
            self.status = VectorStatus(False, f"chromadb query failed: {exc}")
            return []

        ids: list[int] = []
        metadatas = (result.get("metadatas") or [[]])[0]
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            meta_project = metadata.get("project_id")
            if project_id and meta_project not in {project_id, "__global__"}:
                continue
            try:
                ids.append(int(metadata["memory_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if len(ids) >= limit:
                break
        return ids

    def delete_memory(self, memory_id: int) -> bool:
        if not self._ensure_loaded() or self._collection is None:
            return False
        try:
            self._collection.delete(ids=[f"memory:{memory_id}"])
            return True
        except Exception as exc:
            self.status = VectorStatus(False, f"chromadb delete failed: {exc}")
            return False
