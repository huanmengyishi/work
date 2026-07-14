from __future__ import annotations

import atexit
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import regex
from wcwidth import wcswidth

from . import __version__
from .project import Project
from .unicode_text import normalize_unicode_text


COMMANDS = (
    "/new",
    "/resume",
    "/sessions",
    "/status",
    "/undo",
    "/yolo",
    "/super-yolo",
    "/help",
    "/clear",
    "/exit",
)
_GRAPHEME_PATTERN = regex.compile(r"\X")


class ConsoleUI:
    def __init__(
        self,
        project: Project,
        data_dir: Path,
        *,
        yolo: bool = False,
        super_yolo: bool = False,
        show_thinking: bool = True,
        show_reasoning_content: bool = True,
        progress_interval_seconds: int = 10,
        hard_tool_turn_limit: int = 32,
    ) -> None:
        self.project = project
        self.yolo = yolo
        self.super_yolo = super_yolo
        self.show_thinking = show_thinking
        self.show_reasoning_content = show_reasoning_content
        self.progress_interval_seconds = max(1, min(int(progress_interval_seconds), 60))
        self.hard_tool_turn_limit = max(1, min(int(hard_tool_turn_limit), 10_000))
        self.history_path = data_dir / "cache" / "repl_history"
        self.color = bool(sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ)
        self._progress_stop = threading.Event()
        self._progress_thread: threading.Thread | None = None
        self._progress_started = 0.0
        self._progress_label = ""
        self._progress_lock = threading.Lock()
        self._output_lock = threading.RLock()
        self._progress_line_visible = False
        self._reasoning_stream_open = False
        self._readline = self._configure_readline()

    def banner(self) -> None:
        with self._output_lock:
            title = self._style(f"Deep Agent {__version__}", "1;36")
            print(f"\n{title}  |  project: {self.project.name}")
            print(f"workspace: {self.project.root}")
            if self.super_yolo:
                print(self._style("SUPER YOLO enabled: confirmations and permission policies are bypassed.", "1;31"))
            elif self.yolo:
                print(
                    self._style(
                        "YOLO mode enabled: confirmations are skipped; hard safety policies remain active.",
                        "1;31",
                    )
                )
            print(
                "\n首次使用：agent config 查看密钥位置；agent doctor --online 检查连接。\n"
                "快速开始：直接输入任务后按 Enter，例如“总结当前目录文档并生成 Word”。\n"
                "  /help 查看交互用法    /resume 继续未完成任务    /sessions 查看会话\n"
                "  /status 查看状态      /undo 回滚最近修改       Ctrl+C 中断并返回\n"
                "  /yolo on 自动确认普通操作；/super-yolo on 仅用于明确授权的主机级操作\n"
                "示例：\n"
                "  1. 分析当前项目，列出问题并给出有证据的修改建议\n"
                "  2. 修复这个错误，运行测试，完成后再回复\n"
                "  3. 总结所有 Word 文档，生成汇总.docx 并重新打开验证\n"
                "方向键浏览历史，Tab 补全命令。\n"
            )

    def prompt(self, session_id: str | None) -> str:
        session = session_id[-8:] if session_id else "new"
        mode = ":SUPER-YOLO" if self.super_yolo else ":YOLO" if self.yolo else ""
        label = f"{self.project.name}:{session}{mode} > "
        return self._style(label, "1;31" if self.yolo else "1;32", readline_prompt=True)

    def set_yolo(self, enabled: bool) -> None:
        self.yolo = enabled
        if enabled:
            self.info("YOLO mode enabled. Confirmation prompts are disabled; hard safety policies remain active.")
        else:
            self.info("YOLO mode disabled. Confirmation prompts are active.")

    def set_super_yolo(self, enabled: bool) -> None:
        self.super_yolo = enabled
        if enabled:
            self.error("SUPER YOLO enabled. sudo, external paths, privileged Docker, and destructive commands may run.")
        else:
            self.info("SUPER YOLO disabled. Permission Manager hard policies are active.")

    def read(self, session_id: str | None) -> str:
        # Flush before input so WSL terminals always render the prompt before waiting.
        with self._output_lock:
            sys.stdout.flush()
            return normalize_unicode_text(input(self.prompt(session_id))).strip()

    def answer(self, value: str) -> None:
        with self._output_lock:
            self._prepare_block_output_locked()
            print(f"\n{self._style('assistant', '1;36')}\n{normalize_unicode_text(value)}\n")

    def info(self, value: str) -> None:
        with self._output_lock:
            self._prepare_block_output_locked()
            print(self._style(value, "36"))

    def working(self) -> None:
        with self._output_lock:
            self._prepare_block_output_locked()
            print(self._style("正在处理请求...（按 Ctrl+C 可返回交互界面）", "36"), flush=True)
        self.start_progress("分析任务并选择执行方式")

    def start_progress(self, label: str) -> None:
        if not self.show_thinking:
            return
        self.stop_progress(clear=False)
        with self._progress_lock:
            self._progress_label = label
            self._progress_started = time.monotonic()
        self._progress_stop.clear()
        self._progress_thread = threading.Thread(target=self._progress_loop, name="deep-agent-progress", daemon=True)
        self._progress_thread.start()

    def update_progress(self, value: dict) -> None:
        event = str(value.get("event") or "")
        mode = str(value.get("mode") or "standard")
        if event == "strategy.selected":
            strategy = value.get("strategy") if isinstance(value.get("strategy"), dict) else {}
            tier = str(value.get("model_tier") or "standard")
            soft_target = self._progress_count(strategy.get("max_tool_rounds"), default=8)
            hard_limit = max(
                soft_target,
                self._progress_count(
                    value.get("hard_limit"),
                    default=getattr(self, "hard_tool_turn_limit", 32),
                ),
            )
            label = f"{self._mode_label(mode)}；DeepSeek {tier} 档；工具轮次软目标 {soft_target}；硬上限 {hard_limit}"
        elif event == "model.requested":
            step = value.get("current_step") or "当前任务"
            phase = str(value.get("phase") or "tool_loop")
            if phase == "context_compaction":
                label = f"{self._mode_label(mode)}；恢复阶段：语义压缩；工具轮次不增加；步骤 {step}"
            elif phase == "final_synthesis":
                label = f"{self._mode_label(mode)}；收口阶段：最终总结；工具轮次不增加"
            else:
                tool_turn = self._progress_count(value.get("tool_turn", value.get("round")), default=1)
                soft_target = self._progress_count(value.get("soft_target", value.get("max_rounds")), default=8)
                hard_limit = max(
                    soft_target,
                    self._progress_count(
                        value.get("hard_limit"),
                        default=getattr(self, "hard_tool_turn_limit", 32),
                    ),
                )
                label = (
                    f"{self._mode_label(mode)}；工具轮次 {tool_turn}；"
                    f"软目标 {soft_target}；硬上限 {hard_limit}；步骤 {step}"
                )
        elif event == "tool.finished":
            outcome = "完成" if value.get("success") else "失败，正在调整"
            label = f"工具 {value.get('tool', 'unknown')} {outcome}"
        elif event == "context.overflow_recovered":
            stage = {
                "cheap_collapse": "轻量收缩",
                "semantic_compact": "语义压缩",
            }.get(str(value.get("stage") or ""), "上下文收缩")
            estimated_tokens = self._progress_count(value.get("estimated_tokens"), default=0)
            token_suffix = f"；预计上下文 {estimated_tokens} tokens" if estimated_tokens else ""
            label = f"{self._mode_label(mode)}；恢复阶段：{stage}完成；工具轮次不增加{token_suffix}"
        elif event == "model.length_continued":
            attempt = self._progress_count(value.get("attempt"), default=1)
            label = f"{self._mode_label(mode)}；恢复阶段：续写截断输出（第 {attempt} 次）；工具轮次不增加"
        elif event == "context.compacted":
            label = f"{self._mode_label(mode)}；恢复阶段：上下文语义压缩完成；工具轮次不增加"
        elif event == "context.emergency_collapsed":
            label = f"{self._mode_label(mode)}；恢复阶段：紧急上下文收缩完成；工具轮次不增加"
        elif event == "context.compaction_failed":
            circuit = "，熔断已开启" if value.get("circuit_open") else ""
            label = f"{self._mode_label(mode)}；恢复阶段：上下文压缩失败{circuit}；工具轮次不增加"
        elif event == "history.compacted":
            compacted = self._progress_count(value.get("compacted_count"), default=0)
            label = f"{self._mode_label(mode)}；上下文整理：已微压缩 {compacted} 个工具结果；工具轮次不增加"
        elif event == "history.compaction_failed":
            circuit = "，熔断已开启" if value.get("circuit_open") else ""
            label = f"{self._mode_label(mode)}；上下文整理失败{circuit}；工具轮次不增加"
        elif event == "thinking.content":
            if self.show_reasoning_content and bool(value.get("content")):
                self.show_reasoning(str(value["content"]))
            return
        elif event == "thinking.delta":
            content = str(value.get("content") or "")
            if self.show_reasoning_content and content:
                self.show_reasoning_delta(content)
            return
        elif event == "model.responded":
            self.finish_reasoning()
            return
        else:
            return
        # Every non-reasoning progress event is a model/tool boundary. Close a
        # streamed reasoning line before rendering the next status line.
        self.finish_reasoning()
        with self._progress_lock:
            self._progress_label = label
        self._print_progress_line()

    def show_reasoning(self, content: str) -> None:
        text = normalize_unicode_text(content).strip()
        if not text:
            return
        with self._output_lock:
            self._clear_progress_line_locked()
            self._finish_reasoning_locked()
            print(self._style("\n  ● DeepSeek Thinking", "1;35"))
            for line in text.splitlines():
                print(self._style(f"    {line}", "2;35"))
            print(flush=True)

    def show_reasoning_delta(self, content: str) -> None:
        text = normalize_unicode_text(content)
        if not text:
            return
        with self._output_lock:
            if not self._reasoning_stream_open:
                self._clear_progress_line_locked()
                print(self._style("\n  ● DeepSeek Thinking", "1;35"))
                print("    ", end="")
                self._reasoning_stream_open = True
            # Do not clear the line between deltas. A streamed token is a
            # continuation of the open reasoning line, not a new status line.
            print(self._style(text, "2;35"), end="", flush=True)

    def finish_reasoning(self) -> None:
        """Close an open streamed reasoning line at a model/tool boundary."""
        with self._output_lock:
            self._finish_reasoning_locked()

    def _finish_reasoning_locked(self) -> None:
        if self._reasoning_stream_open:
            print(flush=True)
            self._reasoning_stream_open = False

    def stop_progress(self, *, clear: bool = True) -> None:
        thread = getattr(self, "_progress_thread", None)
        stop = getattr(self, "_progress_stop", None)
        if stop is not None:
            stop.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        self._progress_thread = None
        with self._output_lock:
            self._finish_reasoning_locked()
            if clear:
                self._clear_progress_line_locked()

    def _progress_loop(self) -> None:
        self._print_progress_line()
        while not self._progress_stop.wait(1.0):
            self._print_progress_line()

    def _print_progress_line(self) -> None:
        with self._progress_lock:
            elapsed = max(0, int(time.monotonic() - self._progress_started))
            label = self._progress_label
        plain = f"  ◐ Thinking {elapsed}s · {label}"
        if sys.stdout.isatty():
            columns = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns)
            plain = _truncate_display_width(plain, max(1, columns - 1))
        rendered = self._style(plain, "35")
        with self._output_lock:
            # A spinner uses carriage-return replacement. Writing it while a
            # reasoning line is open would erase or overwrite streamed text.
            if self._reasoning_stream_open:
                return
            if sys.stdout.isatty():
                print(f"\r\033[2K{rendered}", end="", flush=True)
                self._progress_line_visible = True
            elif elapsed == 0 or elapsed % self.progress_interval_seconds == 0:
                print(rendered, flush=True)

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            "simple": "轻量直答",
            "standard": "标准工程模式",
            "large": "大规模分块模式",
            "deep": "深度任务图模式",
        }.get(mode, "标准工程模式")

    @staticmethod
    def _progress_count(value: object, *, default: int) -> int:
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if parsed >= 0 else default

    def _clear_progress_line_locked(self) -> None:
        if self._progress_line_visible and sys.stdout.isatty():
            print("\r\033[2K", end="", flush=True)
        self._progress_line_visible = False

    def _prepare_block_output_locked(self) -> None:
        self._clear_progress_line_locked()
        self._finish_reasoning_locked()

    def error(self, value: str) -> None:
        with self._output_lock:
            self._prepare_block_output_locked()
            print(self._style(f"error: {value}", "31"), file=sys.stderr)

    def confirm(self, value: str) -> bool:
        with self._output_lock:
            self._prepare_block_output_locked()
            print(f"\n{self._style('approval required', '1;33')}\n{value}")
            try:
                answer = (
                    normalize_unicode_text(input(self._style("Apply? [y/N] ", "1;33", readline_prompt=True)))
                    .strip()
                    .lower()
                )
            except EOFError:
                print()
                return False
            except KeyboardInterrupt:
                print()
                raise
        return answer in {"y", "yes", "是", "确认"}

    def help(self) -> None:
        with self._output_lock:
            self._prepare_block_output_locked()
            print(
                "\n".join(
                    [
                        "Deep Agent 快速上手（Interactive commands:）",
                        "  直接任务               输入自然语言后按一次 Enter；Agent 会检查、执行、验证后回复",
                        "  Ctrl+C                 中断当前任务并返回交互界面，已完成步骤保存在 Session",
                        "  ↑/↓                    浏览输入历史",
                        "  Tab                    补全斜杠命令",
                        "",
                        "会话命令：",
                        "  /new                   开始新会话",
                        "  /resume [session-id]   选择最近或指定的未完成会话；再次输入继续要求即可执行",
                        "  /sessions              列出保存的会话",
                        "  /status                查看项目、活动会话、预览和快照状态",
                        "",
                        "修改与权限：",
                        "  /undo [snapshot-id]    回滚最近或指定的文件快照",
                        "  /yolo [on|off]         跳过确认，但保留路径和危险操作硬限制",
                        "  /super-yolo [on|off]   绕过权限策略，仅在明确授权主机级操作时使用",
                        "",
                        "界面命令：",
                        "  /clear                 清屏并重新显示快速上手",
                        "  /help                  显示本帮助",
                        "  /exit                  退出 Deep Agent",
                        "",
                        "命令行用法：",
                        "  agent config          查看配置、数据和 API Key 文件位置",
                        '  agent "分析项目并给建议"',
                        '  agent resume --session SESSION_ID "继续完成并验证"',
                        "  agent doctor --online",
                        "  agent --help           查看全部管理命令和启动参数",
                        "",
                        "任务示例：",
                        "  总结当前目录的全部文档，并把汇总写入一个新文件夹",
                        "  分析当前项目，定位失败根因并给出后续修改建议",
                        "  修复现有测试失败，运行验证并说明改动与回滚方法",
                    ]
                )
            )

    def clear(self) -> None:
        with self._output_lock:
            self._finish_reasoning_locked()
            self._progress_line_visible = False
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            else:
                print()
        self.banner()

    def close(self) -> None:
        self.stop_progress()
        if self._readline is None:
            return
        try:
            self._readline.write_history_file(self.history_path)
            self.history_path.chmod(0o600)
        except OSError:
            pass

    def _configure_readline(self):
        try:
            import readline
        except ImportError:
            return None

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(self.history_path)
        except FileNotFoundError:
            pass
        else:
            try:
                self.history_path.chmod(0o600)
            except OSError:
                pass
        readline.set_history_length(1000)
        readline.set_completer(_command_completer)
        readline.parse_and_bind("tab: complete")
        atexit.register(self.close)
        return readline

    def _style(self, value: str, code: str, *, readline_prompt: bool = False) -> str:
        if not self.color:
            return value
        start = f"\033[{code}m"
        end = "\033[0m"
        # GNU Readline counts raw ANSI bytes as printed characters unless they
        # are marked as non-printing. Without these markers its cursor and
        # newline calculations break in some WSL terminals.
        if readline_prompt and self._readline is not None:
            return f"\001{start}\002{value}\001{end}\002"
        return f"{start}{value}{end}"


def _command_completer(text: str, state: int) -> str | None:
    matches = [command for command in COMMANDS if command.startswith(text)]
    return matches[state] if state < len(matches) else None


def _truncate_display_width(value: str, limit: int) -> str:
    """Crop one terminal line without splitting a combining/ZWJ cluster."""

    bounded = max(0, int(limit))
    clusters = list(_grapheme_clusters(value))
    if sum(_cluster_width(item) for item in clusters) <= bounded:
        return value
    ellipsis = "…"
    ellipsis_width = _cluster_width(ellipsis)
    if bounded < ellipsis_width:
        return ""
    selected: list[str] = []
    used = 0
    target = bounded - ellipsis_width
    for cluster in clusters:
        width = _cluster_width(cluster)
        if used + width > target:
            break
        selected.append(cluster)
        used += width
    return "".join(selected) + ellipsis


def _grapheme_clusters(value: str):
    for match in _GRAPHEME_PATTERN.finditer(value):
        yield match.group(0)


def _cluster_width(value: str) -> int:
    # wcwidth accounts for emoji presentation, regional-indicator flags,
    # modifiers, keycaps, ZWJ sequences, CJK width, and zero-width marks.
    # Control strings return -1; they must never expand a terminal layout.
    return max(0, wcswidth(value))
