from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path

from harness2.core import tree
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.runmark import fingerprint
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate.opencode import OPENCODE_SURFACE

from .agent import run_agent
from .files import patch_between, snapshot, text_files
from .trace import Trace

# Reuse the registered Markdown prompt-surface format; execution is LangGraph (lg).
SURFACE = replace(
    OPENCODE_SURFACE,
    plugin_dir=None,
    stubs=(("skills/README.md", "Skills are Markdown procedures loaded with load_skill.\n"),),
)
ALWAYS = set(SURFACE.always_on)


def editable(name):
    return name in ALWAYS or bool(re.fullmatch(r"skills/[a-z0-9][a-z0-9_-]*/SKILL\.md", name))


class LangGraphSubstrate:
    name = "lg"

    def surface(self):
        return SURFACE


DEFAULT_HARNESS = {
    "systemprompt.md": "You are a coding agent. Complete the requested change in the project.\n",
    "guardrails.md": "Preserve unrelated behavior. Do not change existing tests to conceal failures.\n",
    "LongTermMEMORY.md": "",
    "tool_descriptions/tool_guidance.md": "Use filesystem tools to inspect and edit. Use run_command for tests and builds.\n",
    "skills/test-driven-change/SKILL.md": """---
name: test-driven-change
description: implementing or fixing code with executable tests
---
Inspect the relevant implementation and existing tests. Reproduce the failure if possible.
Make the smallest coherent fix. Add a regression test for the reported behavior and run
relevant tests. Distinguish commands you actually ran from checks you recommend.
""",
}


class CodingAdapter:
    def __init__(
        self,
        *,
        root,
        source,
        task,
        model,
        model_id,
        budget,
        trace,
        max_steps=0,
        timeout=0,
        tool_timeout=60,
        harness=None,
        mcp_config=None,
        resume=False,
        stream=False,
        cancel_event=None,
    ):
        self.root, self.source, self.task = root, source, task
        self.model, self.budget, self.trace = model, budget, trace
        self.model_id = model_id
        self.max_steps, self.timeout, self.tool_timeout = max_steps, timeout, tool_timeout
        self.mcp_config = mcp_config
        self.resume = resume
        self.stream, self.cancel_event = stream, cancel_event
        self.substrate = LangGraphSubstrate()
        channel = ChannelSpec("opencode", model_id, timeout=timeout)
        self.spec = BenchSpec(
            name="coding",
            runs_root=root / "engine",
            install_mode="workspace_config",
            prompts=BenchPrompts(
                work="implement and debug software changes in an existing code repository",
                inv_levers="Improve inspection, implementation, testing and verification procedures.",
                dom_levers="Improve procedures for the programming language and project architecture.",
            ),
            channels={"proposer": channel, "selector": channel},
        )
        self.task_id = (
            "task-" + hashlib.sha256(task.encode()).hexdigest()[:10] + "-" + root.name[-8:]
        )
        self.source_files = text_files(source)
        if harness:
            base = tree.read_tree(SURFACE, harness)
            self.validate_tree(base)
        else:
            base = dict(DEFAULT_HARNESS)
        tree.write_tree(SURFACE, base, self.base_harness())
        self.execution_count = 0

    def tasks(self, split="all"):
        return [self.task_id]

    def domain_of(self, task_id):
        return "coding"

    def task_prompt(self, task_id):
        return self.task

    def task_files(self, task_id):
        return self.source

    def base_harness(self):
        return self.root / "base-harness"

    def validate_tree(self, harness):
        unsupported = [name for name in harness if not editable(name)]
        if unsupported:
            raise ValueError("Unsupported LangGraph components: " + ", ".join(unsupported))
        if not harness.get("systemprompt.md", "").strip():
            raise ValueError("systemprompt.md must not be empty")

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        run = self.root / "executions" / run_id
        run.mkdir(parents=True, exist_ok=True)
        signature = fingerprint(harness_dir or self.base_harness())
        metadata = run / "metrics.json"
        if metadata.exists() and self.delivered(run):
            previous = json.loads(metadata.read_text())
            if (
                previous.get("harness_fingerprint") == signature
                and previous.get("model") == self.model_id
            ):
                self.trace.emit("stage", f"CACHE: reusing completed execution {run_id}")
                return run
        workspace = run / "workspace"
        resume_partial = self.resume and workspace.is_dir() and not metadata.exists()
        for name in ("metrics.json", "answer.txt", "changes.patch", "changed-files.json"):
            (run / name).unlink(missing_ok=True)
        if not resume_partial:
            if workspace.exists():
                shutil.rmtree(workspace)
            snapshot(self.source, workspace)
            (run / "state.json").unlink(missing_ok=True)
        active = run / "harness"
        if active.exists():
            shutil.rmtree(active)
        shutil.copytree(harness_dir or self.base_harness(), active)
        label = run_id.rsplit("/", 1)[-1]
        self.execution_count += 1
        self.trace.emit(
            "stage",
            f"SOLVE {label}: "
            + (
                "continuing interrupted workspace"
                if resume_partial
                else "fresh copy of original source; fresh conversation"
            ),
            run_id=run_id,
        )
        trace = Trace(
            run / "trace.jsonl",
            label,
            full=self.trace.full,
            secrets=self.trace.secrets,
            thinking=self.trace.thinking,
        )
        start = time.monotonic()
        task = self.task
        if resume_partial and not (run / "state.json").exists():
            prior = (
                (run / "trajectory.md").read_text()[-24000:]
                if (run / "trajectory.md").exists()
                else ""
            )
            task += (
                "\n\nContinue this interrupted coding attempt. The workspace retains its previous files. "
                "Inspect the files and finish the original task. Previous execution trace (may be truncated):\n"
                + prior
            )
            trace.emit(
                "stage",
                "Older run has no conversation checkpoint; continuing from retained files and execution trace",
            )
        try:
            answer = asyncio.run(
                asyncio.wait_for(
                    run_agent(
                        workspace=workspace,
                        harness=active,
                        task=task,
                        model=self.model,
                        budget=self.budget,
                        trace=trace,
                        max_steps=self.max_steps,
                        stream=self.stream,
                        cancel_event=self.cancel_event,
                        tool_timeout=self.tool_timeout,
                        mcp_config=self.mcp_config,
                        state_file=run / "state.json",
                    ),
                    timeout=self.timeout or None,
                )
            )
        except Exception as exc:
            trace.emit("error", f"Rollout failed: {type(exc).__name__}: {exc}")
            (run / "failure.json").write_text(json.dumps({"error": str(exc)}) + "\n")
            self._artifacts(run, workspace, trace)
            raise
        (run / "answer.txt").write_text(trace.clean(answer) + "\n")
        self._artifacts(run, workspace, trace)
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "wall_s": time.monotonic() - start,
                    "run_id": run_id,
                    "label": label,
                    "completed": True,
                    "harness_fingerprint": signature,
                    "model": self.model_id,
                },
                indent=2,
            )
            + "\n"
        )
        trace.emit("stage", f"SOLVE {label} completed in {time.monotonic() - start:.1f}s")
        return run

    def _artifacts(self, run, workspace, trace):
        patch = trace.clean(patch_between(self.source, workspace))
        (run / "changes.patch").write_text(patch)
        changed = text_files(workspace)
        changed = {
            name: trace.clean(content)
            for name, content in changed.items()
            if self.source_files.get(name) != content
        }
        removed = sorted(set(self.source_files) - set(text_files(workspace)))
        (run / "changed-files.json").write_text(
            json.dumps({"changed": changed, "removed": removed}, indent=2)
        )
        lines = []
        for line in trace.path.read_text().splitlines():
            event = json.loads(line)
            grammar = {
                "tool": "ACTION",
                "result": "OBSERVATION",
                "error": "ERROR",
                "assistant": "ASSISTANT",
                "diff": "DIFF",
            }.get(event["event"])
            if grammar:
                lines.append(
                    f"[{grammar}] {event['message']}\n"
                    + json.dumps(
                        {
                            key: event[key]
                            for key in ("args", "output", "content", "diff")
                            if key in event
                        },
                        ensure_ascii=False,
                    )
                )
        (run / "trajectory.md").write_text("\n\n".join(lines) + "\n")

    def delivered(self, run_dir):
        return (run_dir / "answer.txt").is_file() and (run_dir / "changes.patch").is_file()

    def trajectory(self, run_dir, *, obs_trunc):
        path = run_dir / "trajectory.md"
        return path.read_text() if path.exists() else "(No completed trajectory)"

    def deliverable_text(self, run_dir):
        if not self.delivered(run_dir):
            return "(No completed deliverable)"
        return (
            (run_dir / "answer.txt").read_text()
            + "\n\nCODE CHANGES:\n"
            + (run_dir / "changes.patch").read_text()
        )

    def deliverable_files(self, run_dir):
        if not self.delivered(run_dir):
            return {}
        data = json.loads((run_dir / "changed-files.json").read_text())
        return {
            "answer.txt": (run_dir / "answer.txt").read_text(),
            "changes.patch": (run_dir / "changes.patch").read_text(),
            **{"code/" + name: content for name, content in data["changed"].items()},
        }


class CodingProposer(Proposer):
    def __init__(self, adapter):
        super().__init__(adapter.spec.channels["proposer"])
        self.adapter = adapter

    def call(self, mission, context_dir, label, **kwargs):
        adapter = self.adapter
        context = Path(context_dir)
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", label)
        audit = adapter.root / "improver" / name
        audit.mkdir(parents=True, exist_ok=True)
        trace = Trace(
            audit / "trace.jsonl",
            label,
            full=adapter.trace.full,
            secrets=adapter.trace.secrets,
            thinking=adapter.trace.thinking,
        )
        before = text_files(context / "harness") if (context / "harness").is_dir() else {}
        if label.startswith("experience."):
            stage = "LEARN: retain only lessons supported by observed executions"
        elif label.startswith("select."):
            stage = "SELECT: compare candidate evidence"
        elif label.startswith("inv"):
            stage = "PROPOSE general coding procedure"
        elif label.startswith("dom"):
            stage = "PROPOSE project/domain-specific procedure"
        else:
            stage = "COMPOSE procedures using the trial execution contrasts"
        trace.emit("stage", stage)
        (audit / "mission.md").write_text(trace.clean(mission))
        start = time.monotonic()

        def can_write(path):
            rel = Path(path)
            return (
                len(rel.parts) > 1
                and rel.parts[0] == "harness"
                and editable("/".join(rel.parts[1:]))
            )

        try:
            text = asyncio.run(
                asyncio.wait_for(
                    run_agent(
                        workspace=context,
                        harness=context / "harness",
                        task=mission,
                        model=adapter.model,
                        budget=adapter.budget,
                        trace=trace,
                        max_steps=adapter.max_steps,
                        stream=adapter.stream,
                        cancel_event=adapter.cancel_event,
                        editor=True,
                        writable=can_write,
                        tool_timeout=adapter.tool_timeout,
                    ),
                    timeout=adapter.timeout or None,
                )
            )
        except Exception as exc:
            trace.emit("error", f"Improver failed: {type(exc).__name__}: {exc}")
            raise ProposerError(f"{type(exc).__name__}: {exc}", attempts=1) from None
        text = trace.clean(text)
        (audit / "answer.txt").write_text(text + "\n")
        after = text_files(context / "harness") if (context / "harness").is_dir() else {}
        import difflib

        chunks = []
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                chunks += difflib.unified_diff(
                    before.get(name, "").splitlines(True),
                    after.get(name, "").splitlines(True),
                    fromfile="before/" + name,
                    tofile="after/" + name,
                )
        diff = trace.clean("".join(chunks))
        (audit / "harness.patch").write_text(diff)
        if diff:
            trace.emit("diff", "HARNESS CHANGE (instructions/skills; not project code)", diff=diff)
        trace.emit("stage", f"{stage.split(':')[0]} finished in {time.monotonic() - start:.1f}s")
        reply = Reply(
            label=label,
            text=text,
            ok=True,
            attempts=1,
            wall_s=time.monotonic() - start,
            audit=audit,
            workspace=context / "harness" if kwargs.get("harvest") else None,
        )
        self._calls.append(reply)
        return reply
