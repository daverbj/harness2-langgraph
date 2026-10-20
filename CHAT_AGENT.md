# Interactive coding chat

Follow [the main setup instructions](README.md#setup) first, including creating
`.env` with your OpenRouter key. Start a chat against your own project:

```bash
h2-code chat --repo "/absolute/path/to/your/project" \
  --model minimax/minimax-m3 \
  --test-command "python -m unittest discover -s tests -v"
```

Use your project's actual test command. For a ready-made walkthrough, see
[the small coding demo](examples/coding-demo/README.md).

To import files and a conversation from a previous one-shot run:

```bash
h2-code chat --from-run runs/coding/previous-run
```

The saved model is reused unless overridden. Use `/retry` to continue the
interrupted task. This creates a new chat; reopen an existing chat with
`h2-code chat --session runs/chats/<session-id>` instead.

## Chat normally, improve explicitly

Ordinary messages go to a persistent LangGraph coding assistant:

```text
you ❯ Explain how this project works.
you ❯ Find the cause of the failing tests and fix it.
```

There is no harness search on ordinary messages. The agent can inspect and edit
files, run commands, load skills, and use configured MCP tools when the request
calls for them. Casual questions can be answered without tools.

Use `/improve` to trigger the actual Harness2 engine:

```text
you ❯ /improve "Fix the failing behavior, preserve existing features, and validate with the project tests."
```

Harness2 snapshots the **current chat workspace**, runs a baseline, proposes
instruction/skill changes, probes them in independent workspace copies, composes
a final harness, runs the final coding attempt, and extracts supported lessons.
Each stage has its own label and trace. The improver uses the same streaming
model/tool machinery as the normal chat.

A successful selected result is adopted into the chat workspace along with its
harness. If configured validation fails, the workspace stays unchanged and the
candidate remains available for inspection with `/diff` and explicit `/adopt`.
At k=1, upstream Harness2 adopts its single composition rather than choosing the
best measured test result against baseline. The report distinguishes a real
instruction change from a successful repeated attempt with an unchanged harness.

## Visible execution

The terminal uses colored labels for model requests, assistant text, thinking,
tool calls, code diffs, command output, retries, validation, and Harness2 stages.
Assistant text and command output stream as they arrive. Tool calls execute only
after a complete structured response has been received.

Thinking text or summaries stream **only when the provider exposes them**.
Encrypted continuation blocks are never displayed. Providers that withhold
thinking text still produce elapsed-time progress and usage metadata when
available; the CLI does not manufacture a transcript of hidden reasoning.

`/thinking off` hides thinking text. This changes display/logging, not the model's
reasoning-compute settings. Provider continuation data is retained in private
conversation checkpoints so tool use can continue correctly.

Temporary provider errors (408, 429, 500, 502, 503, 504 and network failures) retry
up to three times with backoff. Errors inside an HTTP-200 SSE stream are handled
too. Incomplete tool calls are never executed. Interrupted text is saved privately
as recovery context. When Gemini supplies a `gemini-continuation` checkpoint,
retries resend it with the unchanged original conversation; otherwise the partial
text is included as tentative notes, not as proof that changes were made.
Completed tools and file edits are retained, with each tool result checkpointed
before starting the next tool in a batch. Ctrl+C records the interrupted tool's
outcome as unknown and leaves remaining calls in that batch unexecuted; `/retry`
returns to the model to inspect and replan instead of replaying the command.
Python tool subprocesses use unbuffered output so progress prints appear promptly.
After retries are exhausted, the chat remains open and `/retry` resumes the interrupted task.
Partial streamed text may appear twice if a response has to be retried.

## Commands

| Input | Action |
|---|---|
| ordinary text | Normal conversational/coding turn |
| `/improve <request>` | Run Harness2 on the current workspace |
| `/retry` | Continue the interrupted conversation or improvement experiment |
| `/model <OpenRouter ID>` | Change the model |
| `/thinking on` / `/thinking off` | Toggle provider-exposed thinking display |
| `/provider Google` / `/provider Google AI Studio` / `/provider auto` | Pin the serving provider and disable fallback, or restore automatic routing |
| `/effort low` / `/effort medium` / `/effort high` / `/effort default` | Set provider reasoning effort; lower effort may reduce latency but affect quality |
| `/tokens 0` | Use the provider's output default without a client token cap |
| `/tokens 131072` | Set an explicit per-response output allowance |
| `/test <command>` | Save and run the project's validation command |
| `/test` | Run the saved validation command |
| `/diff` | Show pending improvement changes, or workspace changes against original |
| `/adopt` | Adopt the last selected candidate, including a failed-validation candidate |
| `/apply` | Copy changes to the original project, with conflict checks and backups |
| `/status` | Show model, paths, limits, pending task and known usage |
| `/clear` | Archive the conversation and start a new one; keep code |
| `/help` | Show commands |
| `/quit` or Ctrl+D | Save and exit |

Enter sends a message. Alt+Enter or Ctrl+J inserts a newline. Pasting multiline
text is supported. Input history and suggestions are kept within the session.
Ctrl+C interrupts active work and returns to the prompt. Shell commands are
terminated as a process group; code and conversation checkpoints are retained.
Interrupted external tools may have partial side effects. Recovery asks the model
to inspect uncertain outcomes; use read-only MCP connections when comparing alternatives.

## Validation and results

Set a validation command that exists and works in your project:

```text
you ❯ /test python -m unittest discover -s tests -v
```

You can also supply `--test-command` at chat startup. `/improve` runs that check
on the original snapshot and all completed candidate workspaces after refinement.
Without a validation command, its report explicitly says improvement is unmeasured.
Use checks appropriate to the change, including manual visual checks for GUI work.

A single task and repeated attempts cannot establish general model quality.
The local reports preserve this distinction; model assertions do not count as
independent proof of improvement.

## Workspaces, applying, and persistence

The original project is copied when the chat starts. Normal coding turns and
adopted improvements change the chat workspace, **not the original directory**.
Use `/apply` when you want those files copied back. It checks for conflicting
external edits before changing any files, supports binary files and deletions,
and saves overwritten files under the session's `backups/` directory. It does
not commit or push Git changes.

The startup banner and `/status` show the session path. Reopen it with:

```bash
h2-code chat --session runs/chats/<session-id>
```

Use `--output runs/my-project-chat` when starting to choose a memorable new session
folder. Do not use an existing output path to create a new session; reopen it
with `--session`. Only one terminal can own a session at a time.

Files in a session include:

- `workspace/`: the current code, build artifacts and adopted changes.
- `original/`: the comparison baseline for the next `/apply`.
- `harness/`: active instructions, memory and skills.
- `conversation.json`: private resumable model/tool messages and continuation data.
- `conversation.stream.json`: private partial-response notes and the latest provider continuation checkpoint, when available. Survives interruption and reopening; removed after a successful response. Old interrupted chat traces can supply visible recovery notes on `/retry`.
- `session.json`: settings, pending operation and known usage; no API key.
- `turns/`: streaming text/thinking and tool traces per chat turn.
- `improvements/`: Harness2 experiments, candidate workspaces and quality reports.
- `library/`: improvement lessons supported by earlier experiments.
- `backups/`: saved files overwritten by adoption or `/apply`.

History and conversation files use owner-only permissions. Thinking text exposed
by the provider is included in traces when thinking display is enabled. Avoid
pasting credentials into the conversation. The configured OpenRouter key is
redacted from streamed output, including when split across token chunks.

## Limits and execution

Chat defaults: no model-turn cap, no total request cap, and no overall duration
limit. `--max-tokens 0` (the default) omits the client output cap; the provider's
own limits still apply. Per-command timeout defaults to 180 seconds, configurable
with `--tool-timeout`. Provider response and stream-idle timeouts are disabled by default, so long
reasoning pauses can continue until you interrupt with Ctrl+C. Connection, write,
and pool acquisition deadlines remain finite. To retry stalled streams after a
chosen interval, reopen with `--stream-idle-timeout 300` (seconds); use
`--stream-idle-timeout 0` to disable it again. Provider-side timeouts such as 504
can still end a request. Positive
`--max-steps` / `--max-calls` opt into limits. `/tokens` changes output allowance
without restarting the session. Truncated responses request continuation.

Shell commands run locally with host/network access; workspace copies are not an
OS security sandbox. Snapshot exclusions and credential handling are described in
[the executor guide](CODING_AGENT.md). MCP configuration uses the same
`--mcp-config` option and stays outside the conversation.

The existing `run`, `improve`, `continue`, and `replay` commands still work.

Request diagnostics show conversation size and message count. Tool names and argument text stream immediately as received under
`TOOL_ARGUMENTS`, before execution; incomplete JSON is expected during generation. Failed requests
record elapsed time, parsed chunk count, argument characters and available provider
name/status in the trace, without logging request contents in these diagnostics.
Large tool calls can be buffered upstream, so silence alone cannot establish
whether generation is progressing. The coding prompt encourages incremental edits
to checkpoint completed work more frequently. Exact recovery of unfinished
provider generation requires a provider checkpoint; text-only recovery cannot
restore hidden computation. Provider-specific checkpoints are dropped when routing
or model settings change, while text recovery context is retained for the same
conversation. Changing providers does not restore the previous provider's hidden
reasoning state.

The waiting message distinguishes waiting for HTTP headers from waiting for the
first or next parsed model chunk. Transport byte counts include SSE keepalives
and framing; they are not token counts. Each request reports first-chunk latency
and the total number of model chunks. The HTTP observer forwards bytes unchanged
and does not decode or collect the response.

To try reducing Argon's reasoning latency when reopening a saved session:

```bash
h2-code chat --session "/path/to/existing/chat" --reasoning-effort low
```

Then use `/retry`. Use `--session` to keep the latest chat workspace and checkpoint;
`--from-run` creates a new chat imported from the older run. The effort option
persists and also applies to `/improve`. `/effort default` restores the provider's
default. `/thinking on` controls visibility only, whereas `/effort` changes the
model request. Streaming cannot eliminate computation time before the provider
sends its first output, or force it to expose individual reasoning tokens.
