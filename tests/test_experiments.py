from __future__ import annotations

import json

from agent.events import Event, EventBus
from agent.experiments import ExperimentRunner


def _config(make_config, *, enabled: bool):
    return make_config(
        {
            "optimizer": {
                "experiments": {
                    "enabled": enabled,
                    "salt": "stable-test",
                    "definitions": [
                        {
                            "id": "convergence-policy",
                            "task_types": ["bug_fix"],
                            "variants": [
                                {
                                    "name": "control",
                                    "weight": 1,
                                    "parameters": {"exploration_round_limit": 6, "ignored": "secret"},
                                },
                                {
                                    "name": "candidate",
                                    "weight": 1,
                                    "parameters": {"exploration_round_limit": 9, "model_tier": "deep"},
                                },
                            ],
                        }
                    ],
                }
            }
        }
    )


def test_experiment_runner_is_default_off_and_assignment_is_stable_and_allowlisted(make_config) -> None:
    assert (
        ExperimentRunner(make_config(), project_id="project-1").assign(
            run_id="run-1",
            task_type="bug_fix",
        )
        is None
    )
    runner = ExperimentRunner(_config(make_config, enabled=True), project_id="project-1")

    first = runner.assign(run_id="run-1", task_type="bug_fix")
    second = runner.assign(run_id="run-1", task_type="bug_fix")

    assert first == second
    assert first is not None
    assert first.experiment_id == "convergence-policy"
    assert first.variant in {"control", "candidate"}
    assert set(first.parameters) <= {"exploration_round_limit", "model_tier"}
    assert runner.assign(run_id="run-1", task_type="review") is None


def test_experiment_outcome_persists_only_scalar_assignment_metadata(tmp_path, make_config) -> None:
    config = _config(make_config, enabled=True)
    runner = ExperimentRunner(config, project_id="project-1")
    runner.path = tmp_path / "experiments" / "project.jsonl"
    events = EventBus()
    runner.attach(events)
    assignment = runner.assign(run_id="run-2", task_type="bug_fix")
    assert assignment is not None
    secret = "PRIVATE_PROMPT_AND_TOKEN"

    events.publish(
        Event(
            "task.finished",
            {
                "prompt": secret,
                "final": secret,
                "state": {
                    "convergence": {"experiment": assignment.to_dict()},
                    "user_request": secret,
                },
            },
            project_id="project-1",
            session_id="session-1",
            run_id="run-2",
        )
    )

    rows = [json.loads(line) for line in runner.path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "experiment_id": "convergence-policy",
            "outcome": "completed",
            "project_id": "project-1",
            "recorded_at": rows[0]["recorded_at"],
            "run_id": "run-2",
            "schema_version": 1,
            "variant": assignment.variant,
        }
    ]
    assert secret not in runner.path.read_text(encoding="utf-8")
