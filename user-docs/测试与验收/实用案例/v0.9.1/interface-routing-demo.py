from __future__ import annotations

from pathlib import Path

from agent.config import AppConfig, DEFAULT_CONFIG
from agent.events import Event, EventBus
from agent.model_router import ModelRouter
from agent.task_plan import TaskPlanFactory
from agent.task_router import TaskRouter


def main() -> None:
    config = AppConfig(values=DEFAULT_CONFIG, config_dir=Path("."), data_dir=Path("."))
    task_router = TaskRouter(config)
    model_router = ModelRouter(config)
    plan_factory = TaskPlanFactory()

    examples = [
        "什么是 Python？",
        "修复这个函数并运行测试",
        "分析整个代码库的所有文件并总结",
        "全面重构生产权限模块并修复所有安全问题",
    ]

    for prompt in examples:
        task = task_router.route(prompt)
        model = model_router.route(task)
        plan = plan_factory.build(task)
        print(
            f"{prompt}\n"
            f"  task={task.task_type}/{task.scale}/{task.risk}/{task.mode}\n"
            f"  deepseek={model.tier} cost={model.cost_class}\n"
            f"  reasons={', '.join(model.reasons)}\n"
            f"  plan={[step['id'] for step in plan]}\n"
        )

    bus = EventBus()
    received: list[str] = []

    def broken(_event: Event) -> None:
        raise RuntimeError("demo subscriber failure")

    def audit(event: Event) -> None:
        received.append(f"{event.name}:{event.effective_run_id}")

    cancel_broken = bus.subscribe("demo.finished", broken)
    bus.subscribe("demo.finished", audit)
    event = bus.publish(
        "demo.finished",
        {"result": "ok"},
        project_id="demo-project",
        session_id="demo-session",
        run_id="demo-session:turn:1",
    )

    print("event=", event.to_dict())
    print("received=", received)
    print("isolated_errors=", bus.last_errors)
    cancel_broken()


if __name__ == "__main__":
    main()
