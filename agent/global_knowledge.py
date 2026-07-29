from __future__ import annotations

from collections.abc import Iterable

from .memory import GLOBAL_KNOWLEDGE_KINDS, MemoryItem, MemoryKind, MemoryStore


class GlobalKnowledgeBase:
    """Facade for reusable, project-independent Memory.

    Only Knowledge, Lesson, and Decision records are visible through this
    boundary. Project records and operational kinds such as Correction,
    Reflection, Bug, and Summary cannot be read or mutated through it.
    """

    ALLOWED_KINDS = GLOBAL_KNOWLEDGE_KINDS

    def __init__(self, store: MemoryStore, *, allow_mutation: bool = False) -> None:
        if not isinstance(allow_mutation, bool):
            raise ValueError("global knowledge allow_mutation must be a boolean")
        self.store = store
        self.allow_mutation = allow_mutation

    def add(
        self,
        *,
        kind: str | MemoryKind,
        title: str,
        content: str,
        tags: Iterable[str] = (),
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> int:
        parsed = self._allowed_kind(kind)
        if not title.strip() or not content.strip():
            raise ValueError("global knowledge title and content must not be empty")
        return self.store.add_memory(
            kind=parsed,
            title=title.strip(),
            content=content.strip(),
            tags=tags,
            project_id=None,
            confidence=confidence,
            expires_at=expires_at,
        )

    def get(self, memory_id: int) -> MemoryItem | None:
        item = self.store.get_memory(memory_id)
        return item if self._is_visible(item) else None

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        kinds: Iterable[str | MemoryKind] | None = None,
        record_usage: bool = True,
    ) -> list[MemoryItem]:
        allowed = self._allowed_kinds(kinds)
        return self.store.search(
            query,
            project_id=None,
            limit=limit,
            global_only=True,
            record_usage=record_usage,
            kinds=allowed,
        )

    def list(
        self,
        *,
        limit: int = 50,
        kinds: Iterable[str | MemoryKind] | None = None,
        tag: str | None = None,
    ) -> list[MemoryItem]:
        allowed = set(self._allowed_kinds(kinds))
        if not allowed:
            return []
        bounded_limit = max(1, min(int(limit), 1000))
        items = self.store.list_memories(
            global_only=True,
            limit=1000,
            tag=str(tag).strip()[:100] if tag else None,
        )
        return [item for item in items if item.kind in allowed][:bounded_limit]

    def update(
        self,
        memory_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: Iterable[str] | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> MemoryItem:
        self._require_mutation_opt_in()
        self._require_visible(memory_id)
        return self.store.update_memory(
            memory_id,
            title=title,
            content=content,
            tags=tags,
            confidence=confidence,
            expires_at=expires_at,
        )

    def delete(self, memory_id: int) -> bool:
        self._require_mutation_opt_in()
        self._require_visible(memory_id)
        return self.store.delete_memory(
            memory_id,
            project_id=None,
            kinds=self.ALLOWED_KINDS,
        )

    @classmethod
    def _allowed_kind(cls, kind: str | MemoryKind) -> MemoryKind:
        parsed = MemoryKind.parse(kind)
        if not isinstance(parsed, MemoryKind) or parsed not in cls.ALLOWED_KINDS:
            allowed = ", ".join(item.value for item in sorted(cls.ALLOWED_KINDS, key=lambda item: item.value))
            raise ValueError(f"global knowledge kind must be one of: {allowed}")
        return parsed

    @classmethod
    def _allowed_kinds(cls, kinds: Iterable[str | MemoryKind] | None) -> tuple[MemoryKind, ...]:
        selected = cls.ALLOWED_KINDS if kinds is None else {cls._allowed_kind(kind) for kind in kinds}
        return tuple(sorted(selected, key=lambda item: item.value))

    @classmethod
    def _is_visible(cls, item: MemoryItem | None) -> bool:
        return bool(
            item is not None and item.project_id is None and item.kind in cls.ALLOWED_KINDS and item.merged_into is None
        )

    def _require_visible(self, memory_id: int) -> MemoryItem:
        item = self.store.get_memory(memory_id)
        if not self._is_visible(item):
            raise ValueError(f"memory {memory_id} is outside the global knowledge boundary")
        assert item is not None
        return item

    def _require_mutation_opt_in(self) -> None:
        if not self.allow_mutation:
            raise PermissionError(
                "global knowledge is add-only by default; construct with allow_mutation=True for explicit maintenance"
            )
