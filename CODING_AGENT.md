# LangGraph coding agent + Harness2

For a streaming terminal conversation with `/improve`, see [interactive chat](CHAT_AGENT.md).

This CLI runs a real LangGraph model/tool loop against a copy of your project.
`run` uses one fixed harness. `improve` uses Harness2's actual Propose–Probe–Compose
engine to modify instructions, memory, and skills and run new coding attempts.
It uses `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from this repository's `.env`.
Choose a model in `.env` or override it per run with `--model`.
The provided `.env.example` uses `minimax/minimax-m3`.

## Start with your own project

Follow [the main setup guide](README.md#setup). If you already have a virtual environment:

```bash
source .venv/bin/activate
python -m pip install -e '.[coding]'
```

Give the agent a repository, a concrete coding task, and a validation command:

```bash
h2-code improve \
  --repo /absolute/path/to/your/project \
  --task "Fix the reported bug, preserve existing behavior, and add a regression test." \
  --test-command "python -m pytest -q"
```

Replace the task with your actual issue and the test command with your project's
command. Dependencies for that project must be installed and its test/build
commands must work without an interactive setup. Tools use the CLI's Python
interpreter; `PYTHONPATH` starts with the copied workspace and its `src/` folder.
Node tooling is available through the inherited PATH, but `node_modules` is not
copied. Projects that need installation can run it with `run_command`.

For a long task, use `--task-file /absolute/path/to/issue.md` instead of `--task`.
A normal single-agent attempt is:

```bash
h2-code run --repo /absolute/path/to/project \
  --task "Your coding task" --test-command "python -m pytest -q"
```

The agent edits a copied workspace. It does not automatically apply changes to
your original repository. The report points to the selected workspace and a
`changes.patch` you can inspect and apply yourself.

For MiniMax M3, set `OPENROUTER_MODEL=minimax/minimax-m3` in `.env`.
You can select it explicitly for a particular run with
`--model minimax/minimax-m3`. Use a fresh output directory when switching models.
The same model currently handles both coding and improvement roles.
The OpenRouter transport preserves provider continuation fields between tool
calls for reasoning models; those fields are not printed in local CLI traces.

## What the terminal shows

Every event has a run label so concurrent proposals and probes remain readable:

- `base1`: ordinary coding agent with the initial instructions.
- `inv1.a1`: propose general coding-procedure improvements from the baseline trace.
- `dom1.a1`: propose improvements specific to this task's language/domain.
- `inv1` / `dom1`: new agent runs that try those instructions, each from original code.
- `c1.a1`: compose instructions based on the trial contrasts.
- `c1`: agent execution under the composed instructions.
- `experience.seq1`: extract evidence-supported lessons for later tasks.
- `harness2`: orchestration, external validation, final comparison and output paths.

You see model turns, tool names and arguments, command output and exit codes,
code diffs, instruction diffs, assistant summaries, errors, elapsed time, and
provider-reported token usage. This is observable execution, not private model
chain-of-thought. Long fields are shortened in the terminal and retained in JSONL.
Pass `--full-trace` to print them without the terminal display limit. Individual
read/tool outputs still have documented size limits to bound model context.

Harness2 internally calls the shared baseline namespace `agg` even for sequential
mode. The executor namespace is `lg`, and the graph is LangGraph. The adapter reuses
the registered OpenCode-style Markdown prompt-surface format; no OpenCode process
is launched. The improver and solver both run LangGraph tool loops.

## Measuring improvement honestly

After the entire workflow, the CLI runs your validation command against:

1. The unmodified source snapshot.
2. The baseline agent's code changes.
3. Each candidate agent's changes.
4. The composed final agent's changes.

The table reports command success/failure and test counts when the runner emits
recognizable pytest/unittest summaries. It prints an improvement, regression,
tie, mixed outcome, or “not measured” statement. A model declaring “better” is
not counted as proof. A zero exit code only checks what the supplied command
actually validates; choose meaningful tests. Build/lint commands can also be used,
but their success is not the same thing as functional correctness.

For a stronger comparison, provide an immutable external test suite. Keep it
outside the source repo so it is not included in the context. It is frozen at
run start and copied into `.h2-eval/` only in temporary evaluation workspaces:

```bash
h2-code improve \
  --repo /absolute/path/to/project \
  --task-file /absolute/path/to/issue.md \
  --test-command "python -m pytest -q" \
  --eval-tests /absolute/path/to/held-out-tests \
  --eval-command "python -m pytest -q .h2-eval"
```

Held-out results are not passed back to Harness2's improver or selector. Public
test results that the coding agent itself obtains through `run_command` do appear
in its trajectory. The agent may add tests; the external suite gives all candidates
the same independent checks. Evidence quality still depends on the suite.

At `--k 1`, upstream Harness2 adopts the single composed final. It does **not**
select the best measured test result against baseline. A regression is explicitly
reported, and failing selected validation produces a nonzero CLI exit status.
No patch is automatically installed. At larger k, the framework selects among
compositions using execution evidence, not held-out correctness results.

One task can demonstrate a measured gain on that task. It cannot establish overall
quality. For that, use many representative tasks, independent tests, repeated
runs, and cost/time comparisons. Refinement uses extra calls; this CLI does not
claim an equal-budget advantage over a longer ordinary-agent run.

## Controls, replay, and outputs

```bash
h2-code improve --repo /path/to/project --task "Actual task" \
  --test-command "python -m pytest -q" \
  --k 1 --mode sequential --max-steps 16 --max-calls 100 \
  --output runs/my-coding-run

# Continue an interrupted run, keeping files and saved conversation state.
# The saved task/model are reused; overall limits are unlimited by default.
h2-code continue runs/my-coding-run

# For reasoning models that exhaust the per-response token budget:
h2-code continue runs/my-coding-run --max-tokens 24000

# No model calls:
h2-code replay runs/my-coding-run
h2-code replay runs/my-coding-run --run c1 --full-trace
```

The default output is a new folder under `runs/coding/`. Request budgets apply
per invocation, including resumed invocations; set an OpenRouter credit limit for
a monetary cap. Defaults: no model-turn limit, no total request limit, and no overall rollout
timeout. The agent runs until it returns a final answer, encounters an error, or
you stop it with Ctrl+C. `--max-steps 0`, `--max-calls 0`, and `--timeout 0` explicitly
select unlimited operation; positive values opt into limits. Per-request limits
remain 6,000 output tokens by default; provider response/read timeouts are disabled.
Connection and write deadlines remain finite. Shell commands have
a 60-second timeout (`--tool-timeout`), and external checks have a 120-second timeout. Responses that reach their output-token limit request continuation instead of
being treated as completed answers. Truncated tool calls are not executed.
There are no automatic SDK retries. The model loop retries transient API failures
up to three times; Harness2 can also retry rejected proposals.

Each output directory contains:

| Path | Contents |
|---|---|
| `report.md`, `report.json` | Baseline/candidate/final quality comparison and metrics |
| `source/` | Frozen original project snapshot |
| `executions/**/workspace/` | Each agent's independent code changes |
| `executions/**/changes.patch` | Code changes relative to original text files |
| `executions/**/state.json` | Private conversation/continuation checkpoint (owner-only permissions) |
| `executions/**/trace.jsonl` | Structured graph, tool, command, usage and diff events |
| `improver/**/harness.patch` | Exact instruction/skill changes from each proposal |
| `improver/**/mission.md` | The actual Harness2 improvement mission |
| `engine/` | Harness2's native stages, selections and reflection evidence |
| `adopted-harness/` | Exported selected instructions and skills |
| `selected-workspace.txt` | Path to selected code workspace |
| `library/` | Evidence-supported improvement lessons, when extracted |
| `trace.jsonl` | Orchestration and final evaluation events |

Text diffs are capped to text files up to 1 MB. Inspect the retained workspace for
larger or binary changes; the patch file is not a complete binary artifact export.
A completed workflow means execution finished, not that quality improved.

## Skills and memory

The active harness has four always-on components:

```text
systemprompt.md
guardrails.md
LongTermMEMORY.md
tool_descriptions/tool_guidance.md
skills/<skill-name>/SKILL.md
```

A `test-driven-change` skill is included. The agent can load it using `load_skill`.
The improver can revise these Markdown files and add valid skill files. Unsupported
script/plugin/subagent changes are rejected rather than pretending they execute.
The graph keeps messages during one rollout; each new trial starts with fresh messages.
Interrupted trials can resume from a private `state.json` checkpoint saved after
each model response and completed tool result. Older runs without checkpoints continue from their existing
workspace and execution trace with a fresh conversation. This does not restart the
project from the original snapshot. Interrupted tool outcomes are recorded as unknown; recovery returns to the model
to inspect and replan rather than blindly replaying the interrupted call.
`LongTermMEMORY.md` is instruction memory. Harness2's library stores adopted
harnesses and supported lessons separately. There is no vector database. Interactive `chat` sessions preserve their conversation
separately; see [the chat guide](CHAT_AGENT.md).

To start from instructions you previously reviewed:

```bash
h2-code run --repo /path/to/project --task "Next task" \
  --harness runs/previous-run/adopted-harness
```

Use `improve --library /absolute/path/to/coding-lessons` to share improvement
lessons across tasks. Reuse it only for appropriate related projects and model
settings; the CLI logs how many earlier entries are available. Cross-task sharing
is opt-in. Tasks use unique IDs so separate runs do not overwrite earlier lessons.

## Optional MCP tools

Pass `--mcp-config /absolute/path/to/mcp.json` using the
`langchain-mcp-adapters` `MultiServerMCPClient` server-map format:

```json
{
  "docs": {
    "transport": "streamable_http",
    "url": "https://docs.langchain.com/mcp"
  }
}
```

MCP tools are added to the solver's graph, and their calls/results use the same
trace events. MCP connections and tool lists stay fixed during refinement;
Harness2 can change their usage guidance. The improver has filesystem tools only,
without shell/MCP access. A conflicting tool name is rejected. MCP server actions
can affect external systems; use read-only tools while comparing candidates so
trial runs do not create repeated external side effects. MCP configuration is not
written into the run config. MCP resumption currently requires a fresh run.

## Execution boundaries

Workspace copies provide comparison isolation, **not an operating-system security
sandbox**. Shell tools can access the host and network. Use trusted tasks and code,
or run the CLI inside a container for stronger isolation. MCP tools follow their
own server permissions.

Snapshotting skips `.git`, virtual environments, caches, `node_modules`, `.env*`,
common key files and symlinks. The output folder is excluded from its own snapshot.
When output is under the source's `runs/`, that generated directory is excluded.
Local file tools reject paths outside their workspace and credential filenames.
The shell still has host access. Child commands omit API-key/token/secret-style
environment variables by default. The configured OpenRouter key is redacted from
saved trace strings. Project source, diffs, and tool results are sent to OpenRouter;
this is not a guarantee that arbitrary secrets embedded in code are removed.

No LangSmith account is required. Terminal traces and local JSONL files work alone.
Optional LangSmith settings inherited from your environment are controlled by the
LangChain SDK.
