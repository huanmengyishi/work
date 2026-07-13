from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from . import __version__, paths
from .config import AppConfig, load_config
from .console import ConsoleUI
from .context import ContextBuilder
from .deepseek import DeepSeekClient
from .daemon import ProjectDaemon
from .memory import MemoryStore
from .network import proxy_url_from_env, redacted_proxy_url
from .parallel import ParallelWorktreeRunner
from .project import Project, ProjectManager, ProjectRegistry
from .runtime import AgentRuntime
from .session import SessionManager
from .task_queue import TaskQueueManager
from .tools import ToolManager
from .tools.docker import DockerTool
from .tools.mcp import MCPManager
from .vector import OptionalChromaStore


COMMANDS = {
    "doctor",
    "projects",
    "init",
    "config",
    "memory",
    "sessions",
    "resume",
    "context",
    "tools",
    "mcp",
    "queue",
    "parallel",
    "health",
    "daemon",
}


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    auto_approve = "--auto-approve" in values
    yolo_flag = "--yolo" in values
    super_yolo_flag = "--super-yolo" in values
    values = [value for value in values if value not in {"--auto-approve", "--yolo", "--super-yolo"}]
    if values and values[0] in {"-h", "--help"}:
        build_help_parser().print_help()
        return 0
    if values and values[0] == "--version":
        print(f"deep-agent {__version__}")
        return 0

    config = load_config()
    yolo = yolo_flag or bool(config.get("permissions.yolo", False))
    super_yolo = super_yolo_flag or bool(config.get("permissions.super_yolo", False))
    if not values:
        return repl(config, auto_approve=auto_approve, yolo=yolo, super_yolo=super_yolo)
    if values[0] == "--":
        prompt = " ".join(values[1:]).strip()
        return (
            run_once(
                config,
                prompt,
                auto_approve=auto_approve,
                yolo=yolo,
                super_yolo=super_yolo,
            )
            if prompt
            else repl(config, auto_approve=auto_approve, yolo=yolo, super_yolo=super_yolo)
        )
    if values[0] not in COMMANDS:
        return run_once(
            config,
            " ".join(values),
            auto_approve=auto_approve,
            yolo=yolo,
            super_yolo=super_yolo,
        )

    parser = build_command_parser()
    args = parser.parse_args(values)
    if args.command == "doctor":
        return cmd_doctor(config, online=args.online, yolo=yolo, super_yolo=super_yolo)
    if args.command == "projects":
        return cmd_projects(config, args.limit)
    if args.command == "init":
        return cmd_init(config)
    if args.command == "config":
        return cmd_config(config)
    if args.command == "memory":
        return cmd_memory(config, args)
    if args.command == "sessions":
        return cmd_sessions(config, args.limit)
    if args.command == "resume":
        return cmd_resume(
            config,
            " ".join(args.prompt),
            args.session,
            auto_approve=auto_approve,
            yolo=yolo,
            super_yolo=super_yolo,
        )
    if args.command == "context":
        return cmd_context(config, args.context_command)
    if args.command == "tools":
        return cmd_tools(config, args.all)
    if args.command == "mcp":
        return cmd_mcp(config, args.mcp_command)
    if args.command == "queue":
        return cmd_queue(
            config,
            args,
            auto_approve=auto_approve,
            yolo=yolo,
            super_yolo=super_yolo,
        )
    if args.command == "parallel":
        return cmd_parallel(config, args, yolo=yolo, super_yolo=super_yolo)
    if args.command == "health":
        return cmd_health(config, args.reset)
    if args.command == "daemon":
        return cmd_daemon(config, args.daemon_command, args.project, once=args.once)
    return 2


def build_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Project-centric DeepSeek CLI agent. Run from any project directory.",
        epilog=(
            'Direct task example: agent "summarize this project"\n'
            "If a task starts with a command name, use: agent -- doctor this code"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Show the installed version.")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve configured snapshot-backed tools (file.apply/file.undo by default).",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Skip all confirmation prompts. Hard permission and path policies remain active.",
    )
    parser.add_argument(
        "--super-yolo",
        action="store_true",
        help="Skip confirmations and Permission Manager hard policies, including sudo restrictions.",
    )
    parser.add_argument("task", nargs="*", help="Natural-language task. No subcommand is required.")
    parser.add_argument(
        "commands",
        nargs="?",
        help="Commands: doctor, projects, init, config, memory, sessions, resume, context, tools, mcp, queue, parallel, health, daemon",
    )
    return parser


def build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="Deep Agent management commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime directories, API key, and dependencies.")
    doctor_parser.add_argument("--online", action="store_true", help="Also send a minimal request to DeepSeek.")
    projects_parser = subparsers.add_parser("projects", help="List registered projects.")
    projects_parser.add_argument("--limit", type=int, default=50)
    subparsers.add_parser("init", help="Initialize and index the current project.")
    subparsers.add_parser("config", help="Show config/data paths and API-key location.")

    memory_parser = subparsers.add_parser("memory", help="Manage long-term memory.")
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_search = memory_sub.add_parser("search", help="Search memory.")
    memory_search.add_argument("query")
    memory_search.add_argument("--global-only", action="store_true")
    memory_search.add_argument("--limit", type=int, default=8)
    memory_add = memory_sub.add_parser("add", help="Add memory.")
    memory_add.add_argument(
        "kind", choices=["Lesson", "Correction", "Reflection", "Bug", "Decision", "Knowledge", "Summary"]
    )
    memory_add.add_argument("title")
    memory_add.add_argument("content")
    memory_add.add_argument("--tag", action="append", default=[])
    memory_add.add_argument("--global-memory", action="store_true")
    memory_list = memory_sub.add_parser("list", help="List recent memory with IDs and short summaries.")
    memory_list.add_argument("--limit", type=int, default=50)
    memory_list.add_argument("--kind")
    memory_list.add_argument("--tag")
    memory_list.add_argument("--global-only", action="store_true")
    memory_delete = memory_sub.add_parser("delete", help="Delete memory by ID.")
    memory_delete.add_argument("id", type=int)
    memory_edit = memory_sub.add_parser("edit", help="Edit memory by ID using flags or $EDITOR.")
    memory_edit.add_argument("id", type=int)
    memory_edit.add_argument("--title")
    memory_edit.add_argument("--content")
    memory_edit.add_argument("--tag", action="append")
    memory_sub.add_parser("stats", help="Show memory totals by scope, kind, and tag.")
    memory_maintain = memory_sub.add_parser("maintain", help="Preview or apply deduplication and expiry cleanup.")
    memory_maintain.add_argument("--apply", action="store_true", help="Apply the reported maintenance operations.")

    sessions_parser = subparsers.add_parser("sessions", help="List resumable sessions for this project.")
    sessions_parser.add_argument("--limit", type=int, default=20)
    resume_parser = subparsers.add_parser("resume", help="Resume the latest or selected session.")
    resume_parser.add_argument("--session", help="Exact session ID or unique prefix. Defaults to latest.")
    resume_parser.add_argument("prompt", nargs="+", help="Continuation request.")

    context_parser = subparsers.add_parser("context", help="Show or rebuild generated project context.")
    context_parser.add_argument("context_command", choices=["show", "refresh", "index"], nargs="?", default="show")
    tools_parser = subparsers.add_parser("tools", help="List registered tool capabilities.")
    tools_parser.add_argument("--all", action="store_true", help="Include disabled capabilities.")
    mcp_parser = subparsers.add_parser("mcp", help="Inspect configured MCP servers and discovered tools.")
    mcp_parser.add_argument("mcp_command", choices=["status", "tools", "config"], nargs="?", default="status")
    queue_parser = subparsers.add_parser("queue", help="Run or resume a persistent serial task queue.")
    queue_parser.add_argument("queue_args", nargs="*")
    queue_parser.add_argument("--id", help="Queue ID or unique prefix for resume/show.")
    queue_parser.add_argument("--continue-on-error", action="store_true")
    parallel_parser = subparsers.add_parser("parallel", help="Run at least 8 independent tasks in Git worktrees.")
    parallel_parser.add_argument("tasks", nargs="+")
    parallel_parser.add_argument("--workers", type=int)
    health_parser = subparsers.add_parser("health", help="Show capability health and broken-tool state.")
    health_parser.add_argument("--reset", nargs="?", const="*", help="Reset one capability or all health failures.")
    daemon_parser = subparsers.add_parser("daemon", help="Manage the optional per-project background daemon.")
    daemon_parser.add_argument(
        "daemon_command", choices=["start", "run", "status", "stop"], nargs="?", default="status"
    )
    daemon_parser.add_argument("--project", help=argparse.SUPPRESS)
    daemon_parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    return parser


def cmd_doctor(
    config: AppConfig,
    *,
    online: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
) -> int:
    vector = OptionalChromaStore(
        Path(str(config.get("memory.vector_path", paths.vector_dir()))).expanduser(),
        enabled=bool(config.get("memory.vector_enabled", True)),
    )
    env_name = str(config.get("model.api_key_env", "DEEPSEEK_API_KEY"))
    key_source = (
        "environment" if os.environ.get(env_name) else "model.yaml" if config.get("model.api_key") else "missing"
    )
    key_status = f"{key_source} ({len(config.api_keys)} key{'s' if len(config.api_keys) != 1 else ''})"
    docker_daemon, docker_proxy = docker_diagnostics()
    rows = [
        ("agent version", __version__),
        ("program dir", str(Path(__file__).resolve().parents[1])),
        ("config dir", str(config.config_dir)),
        ("data dir", str(config.data_dir)),
        ("memory db", str(Path(str(config.get("memory.sqlite_path", paths.memory_db_path()))).expanduser())),
        ("projects db", str(config.data_dir / "projects.db")),
        ("DeepSeek base URL", str(config.get("model.base_url"))),
        ("DeepSeek model", str(config.get("model.model"))),
        (env_name, key_status),
        ("user-space proxy", redacted_proxy_url(proxy_url_from_env())),
        ("git", which("git")),
        ("python", sys.executable),
        ("docker", which("docker")),
        ("docker daemon", docker_daemon),
        ("docker proxy", docker_proxy),
        ("tesseract", which("tesseract")),
        ("pdftotext", which("pdftotext")),
        ("magick", which("magick") or which("convert")),
        ("ai-parser", which("ai-parser")),
        ("pandoc", which("pandoc")),
        ("libreoffice", which("libreoffice")),
        ("node", which("node")),
        ("npx", which("npx")),
        ("chroma", vector.status.reason),
        ("mcp", mcp_diagnostics(config)),
        ("approval mode", "SUPER YOLO" if super_yolo else "YOLO" if yolo else "safe"),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value or 'missing'}")
    if key_source == "missing":
        return 1
    if online:
        try:
            count = DeepSeekClient(config).check_key_pool()
        except Exception as exc:
            print(f"DeepSeek online check  failed: {exc}")
            return 1
        print(f"DeepSeek online check  ready ({count}/{len(config.api_keys)} keys)")
    return 0


def cmd_projects(config: AppConfig, limit: int = 50) -> int:
    registry = ProjectRegistry(config.data_dir / "projects.db")
    rows = registry.list_projects(limit=max(1, limit))
    if not rows:
        print("No registered projects.")
        return 0
    for row in rows:
        tags = ", ".join(json.loads(row["tags"] or "[]"))
        exists = Path(row["root_path"]).exists()
        print(f"{row['last_opened']}  {row['name']}  {row['language'] or '-'}  {'available' if exists else 'missing'}")
        print(f"  id: {row['project_id']}")
        print(f"  root: {row['root_path']}")
        if tags:
            print(f"  tags: {tags}")
    return 0


def cmd_init(config: AppConfig) -> int:
    project, memory = prepare_project(config)
    context = ContextBuilder(config).build(project, refresh=True)
    print(f"Initialized project: {project.name}")
    print(f"Project ID: {project.id}")
    print(f"Root: {project.root}")
    print(f"Context: {project.context_path}")
    print(f"Source index: {context.index_path} ({context.index.get('file_count', 0)} files)")
    return 0


def cmd_config(config: AppConfig) -> int:
    env_name = str(config.get("model.api_key_env", "DEEPSEEK_API_KEY"))
    print(f"config: {config.config_dir}")
    print(f"data: {config.data_dir}")
    print(f"model config: {config.config_dir / 'model.yaml'}")
    print(f"tool config: {config.config_dir / 'tools.yaml'}")
    print(f"main config: {config.config_dir / 'config.yaml'}")
    print(f"API key environment variable: {env_name}")
    print(f"Recommended API key file: {config.config_dir / 'secrets.env'}")
    print(f"Legacy shell environment: {Path.home() / '.bashrc'}")
    return 0


def cmd_memory(config: AppConfig, args: argparse.Namespace) -> int:
    project, memory = prepare_project(config)
    if args.memory_command == "search":
        project_id = None if args.global_only else project.id
        items = memory.search(
            args.query,
            project_id=project_id,
            limit=args.limit,
            global_only=args.global_only,
        )
        if not items:
            print("No memory found.")
            return 0
        for item in items:
            scope = "global" if item.project_id is None else "project"
            print(f"[{item.id}] {scope}/{item.kind}: {item.title}")
            print(item.content.strip()[:1000])
            print()
        return 0
    if args.memory_command == "add":
        project_id = None if args.global_memory else project.id
        tags = list(args.tag)
        if args.kind == "Correction":
            if not any(str(tag).startswith("correction:") for tag in tags):
                print("error: Correction memory requires a correction:<topic> tag", file=sys.stderr)
                return 2
            if project.name not in tags:
                tags.append(project.name)
        memory_id = memory.add_memory(
            kind=args.kind,
            title=args.title,
            content=args.content,
            tags=tags,
            project_id=project_id,
        )
        memory.persist_lesson_file(
            kind=args.kind,
            title=args.title,
            content=args.content,
            project=project,
            global_memory=args.global_memory,
        )
        print(f"Added memory {memory_id}")
        return 0
    if args.memory_command == "list":
        items = memory.list_memories(
            project_id=project.id,
            limit=args.limit,
            kind=args.kind,
            tag=args.tag,
            global_only=args.global_only,
        )
        if not items:
            print("No memory found.")
            return 0
        for item in items:
            scope = "global" if item.project_id is None else "project"
            summary = " ".join(item.content.split())[:180]
            print(
                f"[{item.id}] {item.updated_at}  {scope}/{item.kind}  {item.title} "
                f"confidence={item.confidence:.2f} uses={item.use_count}"
            )
            print(f"  tags: {', '.join(item.tags) or '-'}")
            print(f"  {summary}")
        return 0
    if args.memory_command == "delete":
        item = memory.get_memory(args.id)
        if item is None:
            print(f"error: memory not found: {args.id}", file=sys.stderr)
            return 1
        if not memory.delete_memory(args.id):
            print(f"error: could not delete memory: {args.id}", file=sys.stderr)
            return 1
        print(f"Deleted memory {args.id}: {item.title}")
        return 0
    if args.memory_command == "edit":
        item = memory.get_memory(args.id)
        if item is None:
            print(f"error: memory not found: {args.id}", file=sys.stderr)
            return 1
        title, content, tags = args.title, args.content, args.tag
        if title is None and content is None and tags is None:
            try:
                title, content, tags = edit_memory_interactively(item)
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        if item.kind == "Correction":
            effective_tags = item.tags if tags is None else tags
            if not any(str(tag).startswith("correction:") for tag in effective_tags):
                print("error: Correction memory requires a correction:<topic> tag", file=sys.stderr)
                return 2
        updated = memory.update_memory(args.id, title=title, content=content, tags=tags)
        print(f"Updated memory {updated.id}: {updated.title}")
        return 0
    if args.memory_command == "stats":
        stats = memory.stats(project_id=project.id)
        print(f"total: {stats.total}")
        for label, values in (("scope", stats.by_scope), ("kind", stats.by_kind), ("tag", stats.by_tag)):
            print(f"{label}:")
            for name, count in sorted(values.items(), key=lambda pair: (-pair[1], pair[0].lower())):
                print(f"  {name}: {count}")
        return 0
    if args.memory_command == "maintain":
        report = memory.maintain(project_id=project.id, apply=args.apply)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not args.apply and (report["merge_count"] or report["expired_count"]):
            print("Dry run only. Re-run with --apply to modify memory.")
        return 0
    return 2


def cmd_daemon(config: AppConfig, action: str, project_path: str | None, *, once: bool = False) -> int:
    root = Path(project_path).expanduser() if project_path else Path.cwd()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    daemon = ProjectDaemon(config, project, memory)
    try:
        if action == "start":
            pid = daemon.start()
            print(f"Daemon started for {project.name}: pid={pid}")
            print(f"Log: {daemon.log_path}")
            return 0
        if action == "run":
            return daemon.run(once=once)
        if action == "stop":
            stopped = daemon.stop()
            print("Daemon stopped." if stopped else "Daemon is not running.")
            return 0 if stopped else 1
        status = daemon.status()
        print(f"status: {'running' if status.running else 'stopped'}")
        print(f"project: {status.project_root}")
        print(f"pid: {status.pid or '-'}")
        print(f"state: {daemon.state_path}")
        print(f"log: {daemon.log_path}")
        if status.state.get("last_poll_at"):
            print(f"last poll: {status.state['last_poll_at']}")
        return 0 if status.running else 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def edit_memory_interactively(item) -> tuple[str, str, list[str]]:
    editor = os.environ.get("EDITOR") or shutil.which("nano") or shutil.which("vi")
    if not editor:
        raise RuntimeError("no editor found; set $EDITOR or use --title/--content/--tag")
    payload = {"title": item.title, "content": item.content, "tags": item.tags}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
        path = Path(handle.name)
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    try:
        completed = subprocess.run([editor, str(path)], check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"editor exited with status {completed.returncode}")
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError("edited YAML root must be a mapping")
        tags = value.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError("edited tags must be a YAML list")
        return str(value.get("title") or ""), str(value.get("content") or ""), [str(tag) for tag in tags]
    finally:
        path.unlink(missing_ok=True)


def cmd_sessions(config: AppConfig, limit: int) -> int:
    project = ProjectManager(config).resolve_project(Path.cwd())
    sessions = SessionManager(project).list_sessions(max(1, limit))
    if not sessions:
        print("No saved sessions for this project.")
        return 0
    for item in sessions:
        request = " ".join(item.user_request.split())
        print(f"{item.updated_at}  {item.session_id}  {item.status}  turn={item.turn}")
        print(f"  {request[:160]}")
    return 0


def cmd_resume(
    config: AppConfig,
    prompt: str,
    session_id: str | None,
    *,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
) -> int:
    project, memory = prepare_project(config)
    runtime = build_runtime(
        config,
        project,
        memory,
        auto_approve=auto_approve,
        yolo=yolo,
        super_yolo=super_yolo,
    )
    try:
        print(runtime.resume(prompt, session_id))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_context(config: AppConfig, action: str) -> int:
    project = ProjectManager(config).resolve_project(Path.cwd())
    snapshot = ContextBuilder(config).build(project, refresh=action == "refresh")
    if action == "index":
        print(json.dumps(snapshot.index, ensure_ascii=False, indent=2))
    elif action == "show":
        print(snapshot.rendered)
    else:
        print(f"Refreshed context: {snapshot.generated_path}")
        print(f"Source index: {snapshot.index_path}")
        print(f"Indexed files: {snapshot.index.get('file_count', 0)}")
    return 0


def cmd_tools(config: AppConfig, include_disabled: bool) -> int:
    project, memory = prepare_project(config)
    manager = ToolManager(config, project, memory)
    try:
        for capability in manager.capabilities(enabled_only=not include_disabled):
            if capability.active:
                status = "enabled"
            elif not capability.enabled:
                status = "disabled"
            else:
                status = "unavailable"
            permissions = ",".join(capability.permissions) or "none"
            confirm = " confirm=yes" if capability.requires_confirmation else ""
            print(
                f"{capability.name:<32} {status:<11} model={capability.model_name:<32} "
                f"permissions={permissions} timeout={capability.timeout_seconds}s{confirm}"
            )
            if capability.unavailable_reason and not capability.available:
                print(f"  reason: {capability.unavailable_reason}")
        for status in manager.mcp.statuses:
            if status.error:
                print(f"MCP {status.name}: unavailable ({status.error})")
    finally:
        manager.close()
    return 0


def cmd_health(config: AppConfig, reset: str | None = None) -> int:
    project, memory = prepare_project(config)
    manager = ToolManager(config, project, memory)
    try:
        if reset:
            manager.health.reset(None if reset == "*" else reset)
        rows = manager.health_report()
        for item in rows:
            print(f"{item.name:<40} {item.status:<12} failures={item.consecutive_failures:<3} {item.reason}")
        return 1 if any(item.status == "Broken" for item in rows) else 0
    finally:
        manager.close()


def cmd_mcp(config: AppConfig, action: str) -> int:
    if action == "config":
        print(config.config_dir / "mcp.yaml")
        return 0
    manager = MCPManager(config, Path.cwd())
    try:
        registrations = manager.discover()
        if action == "tools":
            if not registrations:
                print("No active MCP tools. Enable a server in mcp.yaml and set mcp.enabled: true.")
                return 0
            for capability, _ in registrations:
                permissions = ",".join(capability.permissions) or "none"
                confirm = "yes" if capability.requires_confirmation else "no"
                print(
                    f"{capability.model_name:<40} source={capability.name:<40} "
                    f"permissions={permissions} confirm={confirm}"
                )
            return 0

        print(f"MCP: {manager.summary()}")
        if not manager.statuses:
            print("No MCP servers configured.")
        for status in manager.statuses:
            state = "connected" if status.connected else "disabled" if not status.enabled else "unavailable"
            suffix = f" tools={status.tool_count}" if status.connected else ""
            if status.error:
                suffix += f" error={status.error}"
            print(f"- {status.name}: {state}{suffix}")
        return 1 if any(status.enabled and not status.connected for status in manager.statuses) else 0
    finally:
        manager.close()


def cmd_queue(
    config: AppConfig,
    args: argparse.Namespace,
    *,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
) -> int:
    project, memory = prepare_project(config)
    queues = TaskQueueManager(project)
    values = list(args.queue_args)
    command = values.pop(0) if values and values[0] in {"run", "resume", "list", "show"} else "run"
    if command == "list":
        records = queues.list()
        if not records:
            print("No saved queues for this project.")
            return 0
        for record in records:
            completed = sum(task.status == "completed" for task in record.tasks)
            print(f"{record.updated_at}  {record.id}  {record.status}  {completed}/{len(record.tasks)} completed")
        return 0
    if command == "show":
        try:
            record = queues.load(args.id)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
        return 0
    try:
        record = queues.create(values) if command == "run" else queues.load(args.id)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    runtime = build_runtime(
        config,
        project,
        memory,
        auto_approve=auto_approve,
        yolo=yolo,
        super_yolo=super_yolo,
    )

    def runner(task, queue_record) -> tuple[str, str | None, str]:
        plan = []
        for index, item in enumerate(queue_record.tasks):
            dependencies = [queue_record.tasks[index - 1].id] if index > 0 else []
            plan.append(
                {
                    "id": item.id,
                    "title": item.prompt,
                    "description": "Persistent queue task",
                    "dependencies": dependencies,
                    "status": "in_progress" if item.id == task.id else item.status,
                    "retry_count": 0,
                    "max_retries": 1,
                    "allow_parallel": False,
                    "completion_criteria": "The task Session completes successfully.",
                }
            )
        result = runtime.run(task.prompt, initial_plan=plan, queue_id=record.id)
        session_id = runtime.last_session_id
        status = runtime.sessions.load(session_id).state.status if session_id else "failed"
        print(f"[{record.id}] {task.prompt}\n{result}\n")
        return result, session_id, status

    try:
        queues.run(
            record,
            runner,
            stop_on_failure=not args.continue_on_error and bool(config.get("runtime.queue_stop_on_failure", True)),
        )
    except KeyboardInterrupt:
        print(f"Queue paused: {record.id}", file=sys.stderr)
        return 130
    finally:
        runtime.close()
    print(f"Queue {record.id}: {record.status}")
    return 0 if record.status == "completed" else 1


def cmd_parallel(
    config: AppConfig,
    args: argparse.Namespace,
    *,
    yolo: bool = False,
    super_yolo: bool = False,
) -> int:
    project = ProjectManager(config).resolve_project(Path.cwd())
    runner = ParallelWorktreeRunner(project, config.data_dir)
    flags = ["--super-yolo"] if super_yolo else ["--yolo"] if yolo else ["--auto-approve"]
    try:
        run_id, results = runner.run(
            args.tasks,
            min_tasks=int(config.get("runtime.parallel_min_tasks", 8)),
            max_workers=args.workers or int(config.get("runtime.parallel_max_workers", 4)),
            agent_flags=flags,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(
            f"task-{result.index}: {result.status} returncode={result.returncode} "
            f"applied={str(result.applied).lower()} patch={result.patch_path}"
        )
    print(f"Parallel report: {project.agent_dir / 'parallel' / run_id / 'report.json'}")
    return 0 if all(item.status == "completed" and item.applied for item in results) else 1


def repl(
    config: AppConfig,
    *,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
) -> int:
    project, memory = prepare_project(config)
    ui = ConsoleUI(project, config.data_dir, yolo=yolo, super_yolo=super_yolo)
    runtime = build_runtime(
        config,
        project,
        memory,
        approval_handler=lambda request, capability, summary: ui.confirm(summary),
        auto_approve=auto_approve,
        yolo=yolo,
        super_yolo=super_yolo,
    )
    active_session: str | None = None
    ui.banner()
    while True:
        try:
            prompt = ui.read(active_session)
        except (EOFError, KeyboardInterrupt):
            print()
            runtime.close()
            ui.close()
            return 0
        except ValueError as exc:
            ui.error(str(exc))
            continue
        if not prompt:
            ui.info("未输入请求。输入 /help 查看命令，或直接输入任务后按 Enter 执行。")
            continue
        if prompt in {"/exit", "/quit"}:
            runtime.close()
            ui.close()
            return 0
        if prompt == "/help":
            ui.help()
            continue
        if prompt == "/clear":
            ui.clear()
            continue
        if prompt == "/status":
            edit_status = runtime.tools.file_edit.status(active_session)
            ui.info(
                f"Project: {project.name}\nWorkspace: {project.root}\nSession: {active_session or 'new'}\n"
                f"Pending previews: {edit_status['pending_previews']}\n"
                f"Active snapshots: {edit_status['active_snapshots']}\n"
                "Approval mode: "
                + ("SUPER YOLO" if runtime.tools.super_yolo else "YOLO" if runtime.tools.yolo else "safe")
            )
            continue
        if prompt.startswith("/yolo"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                ui.info(f"YOLO mode is {'on' if runtime.tools.yolo else 'off'}.")
                continue
            value = parts[1].strip().lower()
            if value not in {"on", "off"}:
                ui.error("usage: /yolo on|off")
                continue
            runtime.tools.yolo = value == "on"
            ui.set_yolo(runtime.tools.yolo)
            continue
        if prompt.startswith("/super-yolo"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                ui.info(f"SUPER YOLO mode is {'on' if runtime.tools.super_yolo else 'off'}.")
                continue
            value = parts[1].strip().lower()
            if value not in {"on", "off"}:
                ui.error("usage: /super-yolo on|off")
                continue
            runtime.tools.super_yolo = value == "on"
            ui.set_super_yolo(runtime.tools.super_yolo)
            continue
        if prompt.startswith("/undo"):
            if not active_session:
                ui.error("no active session; use /resume first")
                continue
            parts = prompt.split(maxsplit=1)
            result = runtime.tools.file_edit.undo(
                session_id=active_session,
                snapshot_id=parts[1] if len(parts) == 2 else None,
            )
            (ui.info if result.success else ui.error)(result.stdout if result.success else result.stderr)
            continue
        if prompt == "/new":
            active_session = None
            ui.info("Started a new session context.")
            continue
        if prompt == "/sessions":
            cmd_sessions(config, 20)
            continue
        if prompt.startswith("/resume"):
            parts = prompt.split(maxsplit=1)
            try:
                active_session = runtime.sessions.resolve_session_id(parts[1] if len(parts) == 2 else None)
                ui.info(f"Active session: {active_session}")
            except Exception as exc:
                ui.error(str(exc))
            continue
        try:
            ui.working()
            if active_session:
                answer = runtime.resume(prompt, active_session)
            else:
                answer = runtime.run(prompt)
                active_session = runtime.last_session_id
            ui.answer(answer)
        except KeyboardInterrupt:
            active_session = runtime.last_session_id or active_session
            ui.info("请求已中断。可使用 /resume 继续当前会话，或使用 /new 开始新会话。")
        except Exception as exc:
            ui.error(str(exc))


def run_once(
    config: AppConfig,
    prompt: str,
    *,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
) -> int:
    project, memory = prepare_project(config)
    runtime = build_runtime(
        config,
        project,
        memory,
        auto_approve=auto_approve,
        yolo=yolo,
        super_yolo=super_yolo,
    )
    try:
        initial_plan = None
        raw_plan = os.environ.get("DEEP_AGENT_INITIAL_PLAN_JSON")
        if raw_plan:
            try:
                value = json.loads(raw_plan)
                initial_plan = value if isinstance(value, list) else None
            except json.JSONDecodeError:
                initial_plan = None
        print(runtime.run(prompt, initial_plan=initial_plan))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        runtime.close()
    return 0


def prepare_project(config: AppConfig) -> tuple[Project, MemoryStore]:
    project = ProjectManager(config).resolve_project(Path.cwd())
    memory = MemoryStore(config)
    memory.sync_project(project)
    return project, memory


def build_runtime(
    config: AppConfig,
    project: Project,
    memory: MemoryStore,
    *,
    approval_handler=None,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
) -> AgentRuntime:
    tools = ToolManager(
        config,
        project,
        memory,
        approval_handler=approval_handler,
        auto_approve=auto_approve,
        yolo=yolo,
        super_yolo=super_yolo,
    )
    return AgentRuntime(config=config, project=project, memory=memory, tools=tools)


def which(command: str) -> str:
    return shutil.which(command) or "missing"


def docker_diagnostics() -> tuple[str, str]:
    if not shutil.which("docker"):
        return "missing", "missing"
    result = DockerTool(Path.cwd(), timeout=10).run(
        ["info", "--format", "{{.ServerVersion}}|{{.HTTPProxy}}|{{.HTTPSProxy}}"]
    )
    if not result.success:
        return "unavailable", result.stderr or "not configured"
    parts = result.stdout.split("|", 2)
    version = parts[0] if parts else "ready"
    proxies = list(dict.fromkeys(value for value in parts[1:] if value))
    return f"ready ({version})", ", ".join(proxies) if proxies else "not configured"


def mcp_diagnostics(config: AppConfig) -> str:
    if not bool(config.get("mcp.enabled", False)):
        return "disabled"
    manager = MCPManager(config, Path.cwd())
    try:
        manager.discover()
        return manager.summary()
    finally:
        manager.close()
