from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_execution_orchestrator_remains_phase_split() -> None:
    source_root = Path(__file__).parents[1]
    orchestrator_path = source_root / "agent" / "runtime_execution.py"
    tree = ast.parse(orchestrator_path.read_text(encoding="utf-8"))
    mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RuntimeExecutionMixin")
    assert tuple(base.id for base in mixin.bases if isinstance(base, ast.Name)) == (
        "RuntimeExecutionSetupMixin",
        "RuntimeResponseMixin",
        "RuntimeTerminationMixin",
    )

    execute = next(
        node
        for node in mixin.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_execute"
    )
    assert execute.end_lineno is not None
    assert execute.end_lineno - execute.lineno + 1 <= 100
    delegated_calls = {
        node.func.attr
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {
        "_initialize_execution",
        "_request_model_round",
        "_process_model_response",
        "_finish_after_tool_loop",
        "_raise_execution_exception",
    } <= delegated_calls
    assert orchestrator_path.stat().st_size <= 12_000

    for module_name in (
        "runtime_execution_setup.py",
        "runtime_response.py",
        "runtime_termination.py",
    ):
        module_path = source_root / "agent" / module_name
        assert module_path.is_file()
        assert module_path.stat().st_size <= 25_000
