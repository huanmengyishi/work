# Deep Agent V3

DeepSeek-only project agent for WSL/Linux. The GitHub tree contains only the
installable runtime and its minimal launch entry; tests, CI, development
scripts, detailed reports, Word files, and private runtime data are kept local.

Current release: `0.14.1` · AgentState schema: `8` · core interface contract: `6`

## Install

```bash
git clone https://github.com/huanmengyishi/work.git ~/AI-Agent
cd ~/AI-Agent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/agent --version
```

Optional capabilities are installed explicitly:

```bash
.venv/bin/python -m pip install -e '.[browser,semantic,document,vector]'
```

## Configure the DeepSeek key

Keep the key outside the repository:

```bash
mkdir -p ~/.config/deep-agent
nano ~/.config/deep-agent/secrets.env
chmod 600 ~/.config/deep-agent/secrets.env
```

```text
DEEPSEEK_API_KEY=replace_with_your_valid_key
```

Multiple keys may be separated by English or Chinese commas. Never commit this
file or copy keys, Memory, Sessions, logs, browser state, caches, or
`.project-agent` data into the source tree.

## Start and use

```bash
cd /path/to/project
~/AI-Agent/.venv/bin/agent init
~/AI-Agent/.venv/bin/agent
~/AI-Agent/.venv/bin/agent "inspect this project and verify the requested change"
```

Useful commands:

```bash
agent --help
agent --version
agent doctor
agent sessions
agent resume
agent context show
agent tools --all
agent health
agent memory search "query"
```

Interactive input is submitted with one `Enter`. Empty input has explicit
feedback, `Ctrl+C` returns to a recoverable prompt, and `/resume` continues a
saved Session. `/vector-retry` performs one explicit lazy ChromaDB retry after
the optional dependency has been installed or repaired.

## Large documents and safe file changes

Large-document generation uses an approved outline, one durable chapter at a
time, and process-independent checkpoints. Word output must follow:

```text
document_generator_render -> file_apply -> document_parse -> document_generator_finalize
```

Completed chapter bodies are removed from later model context after their
checkpoint hashes are verified. File mutations use the snapshot-backed
`file_diff -> file_apply -> file_undo` flow.

## Configuration defaults

- DeepSeek is the only inference provider.
- Heavy vector, adaptive, and experimental features remain disabled by default.
- Configuration migration adds missing defaults and does not replace an
  existing user choice.
- The task wall clock defaults to one hour and all model/tool/network inputs
  retain bounded size, count, timeout, and output limits.
- `--super-yolo` relaxes permission policy and should be treated as unsafe.

Configuration is stored under `~/.config/deep-agent`; runtime data is stored
under `~/.local/share/deep-agent`; each managed project uses a private
`.project-agent` directory.

## Offline verification

These checks do not call DeepSeek:

```bash
agent --version
agent --help
agent doctor
```

`agent doctor --online` makes real DeepSeek requests and may exercise multiple
configured keys. Use it only when an online credential check is intentional.

## Risk and rollback

Version `0.14.1` keeps AgentState schema `8`, but older releases may not
understand newer workflow evidence. Preserve active Sessions and never delete
Memory or project data during rollback.

```bash
cd ~/AI-Agent
git switch --detach v0.13.0
.venv/bin/python -m pip install -e .
```

Return to the current `main`/tag and reinstall the package to restore the latest
runtime.
