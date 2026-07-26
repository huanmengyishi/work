from __future__ import annotations

from types import SimpleNamespace

from agent import cli
from agent.deepseek import DeepSeekClient
from agent.memory import MemoryStore
from agent.vector import OptionalChromaStore, VectorStatus


def test_enabled_vector_store_loads_only_on_first_real_operation(tmp_path, monkeypatch) -> None:
    loads: list[str] = []

    def fake_load(store: OptionalChromaStore) -> VectorStatus:
        loads.append(str(store.path))
        return VectorStatus(False, "test backend unavailable")

    monkeypatch.setattr(OptionalChromaStore, "_load", fake_load)
    store = OptionalChromaStore(tmp_path / "vector", enabled=True)

    assert store.is_enabled()
    assert "loads on first vector operation" in store.status.reason
    assert loads == []
    assert store.query_memory_ids(query="   ", project_id=None, limit=5) == []
    assert loads == []

    assert not store.upsert_memory(
        memory_id=1,
        project_id=None,
        kind="Lesson",
        title="Lazy",
        content="Do not initialize Chroma during CLI startup.",
        tags=[],
    )
    assert loads == [str(tmp_path / "vector")]
    assert not store.is_enabled()

    # A failed optional backend is not repeatedly imported or initialized.
    assert store.query_memory_ids(query="retry", project_id=None, limit=5) == []
    assert loads == [str(tmp_path / "vector")]


def test_no_key_direct_resume_and_doctor_never_load_old_enabled_vector(make_config, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = make_config({"memory": {"vector_enabled": True}})

    def unexpected_load(_store: OptionalChromaStore) -> VectorStatus:
        raise AssertionError("API-key preflight and doctor must not initialize Chroma")

    def unexpected_prepare(_config):
        raise AssertionError("direct and resume must fail before project/Memory setup")

    monkeypatch.setattr(OptionalChromaStore, "_load", unexpected_load)
    monkeypatch.setattr(cli, "prepare_project", unexpected_prepare)
    monkeypatch.setattr(cli, "docker_diagnostics", lambda: ("not checked", "not checked"))
    monkeypatch.setattr(cli, "mcp_diagnostics", lambda _config: "not checked")

    assert cli.run_once(config, "do work") == 1
    assert cli.cmd_resume(config, "continue", None) == 1
    assert cli.cmd_doctor(config) == 1

    captured = capsys.readouterr()
    assert "loads on first vector operation" in captured.out
    assert "secrets.env" in captured.err


def test_no_key_repl_task_never_loads_old_enabled_vector(make_config, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = make_config({"memory": {"vector_enabled": True}})
    memory = MemoryStore(config)
    project = SimpleNamespace(name="test", root=tmp_path)
    errors: list[str] = []
    prompts = iter(["do work", "/exit"])

    def unexpected_load(_store: OptionalChromaStore) -> VectorStatus:
        raise AssertionError("a keyless REPL task must not initialize Chroma")

    class FakeUI:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def banner(self) -> None:
            pass

        def read(self, _session_id):
            return next(prompts)

        def error(self, message: str) -> None:
            errors.append(message)

        def update_progress(self, *_args, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeRuntime:
        client = DeepSeekClient(config)

        @staticmethod
        def close() -> None:
            pass

    monkeypatch.setattr(OptionalChromaStore, "_load", unexpected_load)
    monkeypatch.setattr(cli, "prepare_project", lambda _config: (project, memory))
    monkeypatch.setattr(cli, "ConsoleUI", FakeUI)
    monkeypatch.setattr(cli, "build_runtime", lambda *_args, **_kwargs: FakeRuntime())

    assert cli.repl(config) == 0
    assert errors and "secrets.env" in errors[0]
    assert not (tmp_path / "data" / "vector").exists()
