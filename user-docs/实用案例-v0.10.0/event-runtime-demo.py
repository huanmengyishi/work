from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent.config import AppConfig, DEFAULT_CONFIG, deep_merge
from agent.event_pipelines import EventMetricsCollector, MemoryUsageEventPipeline
from agent.events import Event, EventBus, EventDispatchError, JsonlEventLogger
from agent.memory import MemoryStore


def config_for(root: Path) -> AppConfig:
    values = deep_merge(
        DEFAULT_CONFIG,
        {
            "memory": {
                "sqlite_path": str(root / "memory.db"),
                "vector_path": str(root / "vector"),
                "vector_enabled": False,
            }
        },
    )
    return AppConfig(values=values, config_dir=root / "config", data_dir=root / "data")


def main() -> None:
    private_marker = "PRIVATE-PROMPT-REASONING-TOOL-OUTPUT"
    with tempfile.TemporaryDirectory(prefix="deep-agent-event-demo-") as temporary:
        root = Path(temporary)
        config = config_for(root)
        memory = MemoryStore(config)
        memory_id = memory.add_memory(
            kind="Lesson",
            title="Idempotent context usage",
            content="The same usage event must reinforce this item only once.",
            project_id="demo-project",
        )

        events = EventBus()
        persisted: list[str] = []
        events.subscribe(
            "session.checkpoint.requested",
            lambda event: persisted.append(str(event.run_id)),
            required=True,
            name="demo.session-writer",
        )
        events.subscribe(
            "session.checkpoint.requested",
            lambda _event: (_ for _ in ()).throw(RuntimeError("optional metrics unavailable")),
            name="demo.optional-observer",
        )
        MemoryUsageEventPipeline(memory, events)

        audit = JsonlEventLogger(root / "audit")
        metrics = EventMetricsCollector(root / "metrics" / "demo.json")
        events.subscribe("*", audit, name="demo.audit")
        events.subscribe("*", metrics, name="demo.metrics")

        dispatch = events.dispatch_required(
            "session.checkpoint.requested",
            {
                "state": {"prompt": private_marker},
                "messages": [{"reasoning_content": private_marker}],
            },
            project_id="demo-project",
            session_id="demo-session",
            run_id="demo-session:turn:1",
        )
        assert persisted == ["demo-session:turn:1"]
        assert dispatch.required_errors == ()
        assert len(dispatch.errors) == 1

        usage = {
            "memory_ids": [memory_id, memory_id],
            "usage_id": "demo-session:turn:1:context-package:memory-1",
        }
        for _ in range(2):
            events.dispatch_required(
                "memory.usage.recorded",
                usage,
                project_id="demo-project",
                session_id="demo-session",
                run_id="demo-session:turn:1",
            )
        assert memory.get_memory(memory_id).use_count == 1

        events.publish(
            Event(
                "tool.finished",
                {
                    "request": {"capability": "demo.run", "args": {"secret": private_marker}},
                    "result": {
                        "success": False,
                        "duration_ms": 23,
                        "stdout": private_marker,
                        "stderr": private_marker,
                    },
                    "prompt": private_marker,
                    "reasoning": private_marker,
                },
                project_id="demo-project",
                session_id="demo-session",
                run_id="demo-session:turn:1",
            )
        )

        audit_path = next((root / "audit").glob("events-*.jsonl"))
        audit_text = audit_path.read_text(encoding="utf-8")
        metrics_value = json.loads((root / "metrics" / "demo.json").read_text(encoding="utf-8"))
        assert private_marker not in audit_text
        assert private_marker not in json.dumps(metrics_value)
        assert metrics_value["counts"] == {"tool.finished": 1}
        assert metrics_value["total_tool_duration_ms"] == 23
        assert metrics_value["failed_tools"] == 1

        missing = EventBus()
        try:
            missing.dispatch_required("session.finalize.requested", {})
        except EventDispatchError as exc:
            print(f"Missing required owner correctly failed closed: {exc.event_name}")
        else:
            raise AssertionError("missing required owner did not fail closed")

        print("Safe audit sample:")
        print(audit_text.splitlines()[-1])
        print("Metrics:")
        print(json.dumps(metrics_value, ensure_ascii=False, indent=2))
        print("Event Runtime demo passed.")


if __name__ == "__main__":
    main()
