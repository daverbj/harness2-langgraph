# Harness² + LangGraph Coding Agent

A streaming terminal coding assistant with OpenRouter models and an explicit
`/improve` command for Harness² refinement. Maintained by
[daverbj](https://github.com/daverbj), built on
[Google Research's Harness²](https://github.com/google-research/harness2).

Chat normally to discuss or change code. Use `/improve` when you want Harness² to
run a baseline, propose changes to the agent's instructions and skills, probe
those changes, compose a refined harness, and attempt the task again. Model
weights stay fixed. Refinement makes extra model calls and does not guarantee
better code; configure independent tests to measure the result.

This is a community extension, not an official Google product. The original
framework, benchmark documentation, attribution, and history are retained. See
[the upstream README](UPSTREAM_README.md) for the original research project.

## Setup

Requirements: **macOS or Linux, Git, Python 3.12 or 3.13**, and an OpenRouter API
key with access to your selected model. Install your target project's build and
test dependencies separately. Docker, QEMU, Google Cloud, and Node.js are needed
only when your particular project or an upstream benchmark requires them.

```bash
git clone https://github.com/daverbj/harness2-langgraph.git
cd harness2-langgraph
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[coding]'
cp .env.example .env
```

Edit `.env` and fill in your key:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=minimax/minimax-m3
```

The real `.env`, virtual environments, and `runs/` directory are ignored by Git.
Run the CLI from this checkout to use its `.env`. Exported shell variables take
precedence. With `uv`, you can instead use `uv venv --python 3.12 .venv` and
`uv pip install -e '.[coding]'` after activating the environment.

## First run: a small coding task

The [demo project](examples/coding-demo/README.md) has a deliberately broken
`clamp()` function and five tests. Its tests initially fail by design.

```bash
h2-code chat \
  --repo examples/coding-demo \
  --output runs/my-first-chat \
  --test-command "python -m unittest discover -s tests -v"
```

At the `you ❯` prompt, enter:

```text
Explain the project and its failing tests.
Fix clamp() so that all existing tests pass. Run the tests and summarize the changes.
/diff
/test
```

The agent works in `runs/my-first-chat/workspace/`, a copy of the demo. To copy
reviewed changes back to the original demo, enter `/apply`. You can inspect the
workspace directly without applying it.

## Use your own project

```bash
h2-code chat \
  --repo "/absolute/path/to/your/project" \
  --model minimax/minimax-m3 \
  --test-command "python -m pytest -q"
```

Use the test command that actually works for your project. Ordinary messages use
the persistent LangGraph model/tool loop. To invoke real Harness² refinement:

```text
/improve "Fix the failing behavior, preserve existing behavior, and validate with the project's tests."
```

`/improve` snapshots the current chat workspace and explores harness changes in
separate workspaces. A completed selected result is adopted into the chat when
configured validation passes. Failed-validation candidates remain available for
review and explicit `/adopt`. Without a validation command, quality improvement
is reported as unmeasured. At `k=1`, selection follows upstream Harness²'s
composition policy, rather than picking whichever candidate has the best test
score. See [the chat guide](CHAT_AGENT.md) for the full behavior.

## Chat commands

| Command | Purpose |
| --- | --- |
| `/improve <request>` | Run Harness² refinement for a coding task |
| `/test [command]` | Set/run project validation |
| `/diff` | Inspect changes |
| `/apply` | Export workspace changes to the original project, with conflict checks |
| `/model minimax/minimax-m3` | Change the OpenRouter model |
| `/provider auto` | Let OpenRouter route requests |
| `/provider Google` | Pin a provider and disable fallback; the provider must serve your model |
| `/effort low` | Set reasoning effort where the selected model supports it |
| `/thinking on` or `/thinking off` | Show/hide provider-exposed reasoning text |
| `/tokens 0` | Use the provider's output-token default |
| `/status` | Show session paths, settings, and known token usage |
| `/retry` | Recover an interrupted task |
| `/clear` | Archive conversation history and keep workspace files |
| `/quit` | Save and exit |

Enter sends a message; Alt+Enter inserts a newline. Ctrl+C interrupts active
work. Reopen the **same session** to retain its latest files and conversation:

```bash
h2-code chat --session runs/my-first-chat
```

`--from-run /path/to/older/run` imports an earlier one-shot run into a **new**
chat; it does not reopen an existing chat.

## Streaming, recovery, and limits

Assistant text, provider-exposed reasoning, tool arguments, and command output
stream as received. A provider may buffer output, withhold reasoning, or send
several tokens in one chunk; the client cannot force individual-token delivery.
Network keepalives are reported separately from model chunks.

Completed tools and workspace edits are checkpointed. Interrupted text is kept
as tentative recovery context. Gemini continuation checkpoints can resume
interrupted generation when the provider supplies them; exact hidden reasoning
cannot be reconstructed without one. Interrupted commands return to the model
for inspection and replanning instead of blindly replaying.

Chat defaults have no total request, turn, or overall duration cap. Response/read
idle timeouts are disabled by default, while connection/write deadlines remain
finite. Shell commands have a 180-second timeout. Provider-side 504 errors can
still end a request. Use `--max-calls`, `--max-steps`, `--max-tokens`,
`--tool-timeout`, and `--stream-idle-timeout` to configure limits. API calls may
incur charges, including retries.

Filesystem tools restrict paths to the copied workspace, and child shell commands
omit API credential variables. **Shell and MCP tools still run with your host's
permissions; a workspace copy is not a security sandbox.** Session traces can
contain project code and provider reasoning. Keep them private unless reviewed.

## One-shot execution, skills, memory, and MCP

```bash
# One coding attempt with a fixed harness
h2-code run --repo /path/to/project --task "Fix the bug" \
  --test-command "python -m pytest -q"

# Harness² refinement with external validation
h2-code improve --repo /path/to/project --task "Fix the bug" \
  --test-command "python -m pytest -q" --k 1 --mode sequential

# Reopen an interrupted one-shot run, or replay its saved trace
h2-code continue runs/coding/your-run
h2-code replay runs/coding/your-run
```

One-shot runs default to a 6,000-token response allowance and a 60-second shell
command timeout; chat uses provider-default output allowance and 180 seconds.

The active harness contains a system prompt, guardrails, tool guidance,
`LongTermMEMORY.md`, and loadable `skills/<name>/SKILL.md` files. Harness²'s
library retains adopted harnesses and supported lessons. Optional MCP servers
are configured with `--mcp-config /path/to/mcp.json`. See
[the executor guide](CODING_AGENT.md) for configuration, held-out evaluation,
reports, and supported harness components.

## Development

```bash
python -m pip install -e '.[coding,render]'
python -m unittest discover -s tests/coding -v
python -m unittest discover -s tests -v
```

The coding tests use deterministic fake models and a local HTTP streaming server;
they do not require a real API key. For a credential-free upstream engine demo:

```bash
harness2-demo --output runs/engine-demo --k 1
```

Contributions should include a reproducible case and relevant verification. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License and attribution

[Apache License 2.0](LICENSE). Google LLC retains copyright in the upstream
Harness² work. This repository adds the LangGraph/OpenRouter coding CLI,
interactive chat, recovery, and related examples and documentation. See
[NOTICE](NOTICE), [the upstream README](UPSTREAM_README.md), and
[the upstream setup guide](docs/SETUP.md).
