from __future__ import annotations

import json
from pathlib import Path

from agent.context import ContextBuilder
from agent.project import ProjectManager, ProjectRegistry


def test_context_builder_indexes_project_and_reuses_cache(tmp_path: Path, make_config) -> None:
    root = tmp_path / "sample"
    root.mkdir()
    (root / "README.md").write_text("# Sample\n\nRuntime notes.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    (root / "main.py").write_text(
        "class Worker:\n    pass\n\ndef run():\n    return 1\n",
        encoding="utf-8",
    )
    config = make_config()
    project = ProjectManager(config).resolve_project(root)

    first = ContextBuilder(config).build(project, refresh=True)
    second = ContextBuilder(config).build(project)

    assert project.language == "Python"
    assert first.index["entry"] == "main.py"
    assert first.index["file_count"] == 3
    assert {item["name"] for item in first.index["symbols"]} == {"Worker", "run"}
    assert "README.md" in first.loaded_files
    assert second.index["cache_hit"] is True
    assert second.index_path.exists()
    assert second.generated_path.exists()


def test_project_move_keeps_uuid_and_updates_registry(tmp_path: Path, make_config) -> None:
    config = make_config()
    original = tmp_path / "old-name"
    original.mkdir()
    first = ProjectManager(config).resolve_project(original)

    moved = tmp_path / "new-name"
    original.rename(moved)
    second = ProjectManager(config).resolve_project(moved)
    rows = ProjectRegistry(config.data_dir / "projects.db").list_projects()

    assert second.id == first.id
    assert second.name == "new-name"
    assert len(rows) == 1
    assert rows[0]["root_path"] == str(moved.resolve())
    assert rows[0]["name"] == "new-name"


def test_project_root_ignores_empty_git_directory(tmp_path: Path, make_config) -> None:
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    root = outer / "project"
    root.mkdir()

    project = ProjectManager(make_config()).resolve_project(root)

    assert project.root == root.resolve()
    assert project.name == "project"


def test_optional_semantic_index_is_sidecar(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "service.py").write_text(
        "import json\n\nclass UserService:\n    def load(self, user_id: int):\n        return user_id\n",
        encoding="utf-8",
    )
    config = make_config({"context": {"semantic_index_enabled": True, "semantic_languages": ["python"]}})
    project = ProjectManager(config).resolve_project(root)
    snapshot = ContextBuilder(config).build(project, refresh=True)

    semantic_path = project.agent_dir / "index.semantic.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    assert snapshot.index_path.name == "index.json"
    assert semantic["enabled"] is True
    assert semantic["files"][0]["path"] == "service.py"
    assert semantic["files"][0]["structures"][0]["name"] == "UserService"
    assert semantic["files"][0]["imports"][0]["source"] == "import json"
    assert "UserService.load" in snapshot.rendered
