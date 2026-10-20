"""A LangGraph coding agent with visible Harness2 refinement and external checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from harness2 import config  # Loads the repository .env without overriding shell variables.
from harness2.core.library import Library
from harness2.core.schedule import mode_token, run_task

from .adapter import SURFACE, CodingAdapter, CodingProposer
from .agent import Budget, make_model
from .files import snapshot
from .report import build_report
from .trace import CONSOLE, Trace, reliable_output, show_event


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "improve"):
        p = sub.add_parser(
            action,
            help="Run a coding task" if action == "run" else "Run actual Harness2 refinement",
        )
        p.add_argument("--repo", required=True, type=Path, help="Project to copy and edit")
        task = p.add_mutually_exclusive_group(required=True)
        task.add_argument("--task", help="Coding task in plain language")
        task.add_argument("--task-file", type=Path)
        p.add_argument("--model", default=os.getenv("OPENROUTER_MODEL") or "minimax/minimax-m3")
        p.add_argument(
            "--output", type=Path, help="Output directory; default runs/coding/<unique-id>"
        )
        p.add_argument(
            "--resume",
            action="store_true",
            help="Resume an existing output with the same task/settings",
        )
        p.add_argument(
            "--harness", type=Path, help="Use an existing base/adopted harness directory"
        )
        p.add_argument(
            "--test-command", help="Project tests; run externally on every final workspace"
        )
        p.add_argument(
            "--eval-command", help="Independent held-out check, run only after refinement"
        )
        p.add_argument(
            "--eval-tests",
            type=Path,
            help="External test file/directory; copied to .h2-eval only during evaluation",
        )
        p.add_argument(
            "--max-steps",
            type=int,
            default=0,
            help="Model turns per solve/proposal; 0 = unlimited (default)",
        )
        p.add_argument(
            "--max-calls",
            type=int,
            default=0,
            help="Total attempted model requests; 0 = unlimited (default)",
        )
        p.add_argument(
            "--max-tokens", type=int, default=6000, help="Output token limit per model request"
        )
        p.add_argument(
            "--timeout",
            type=int,
            default=0,
            help="Seconds per solve/proposal; 0 = unlimited (default)",
        )
        p.add_argument("--tool-timeout", type=int, default=60)
        p.add_argument("--check-timeout", type=int, default=120)
        p.add_argument(
            "--mcp-config", type=Path, help="MultiServerMCPClient JSON configuration (optional)"
        )
        p.add_argument(
            "--full-trace", action="store_true", help="Print full tool results and diffs"
        )
        if action == "improve":
            p.add_argument("--k", type=int, default=1)
            p.add_argument("--mode", choices=("sequential", "parallel"), default="sequential")
            p.add_argument(
                "--library", type=Path, help="Persist improvement lessons across coding tasks"
            )
    chat = sub.add_parser("chat", help="Interactive streaming coding chat; /improve runs Harness2")
    chat.add_argument("--repo", type=Path)
    chat.add_argument("--session", type=Path, help="Reopen a saved chat")
    chat.add_argument(
        "--from-run", type=Path, help="Import files and conversation from an interrupted run"
    )
    chat.add_argument("--output", type=Path, help="New chat session directory")
    chat.add_argument("--model")
    chat.add_argument(
        "--provider", help="OpenRouter provider name; pins routing and disables fallback"
    )
    chat.add_argument("--harness", type=Path)
    chat.add_argument(
        "--max-tokens", type=int, help="Output tokens per response; 0 uses provider defaults"
    )
    chat.add_argument(
        "--stream-idle-timeout",
        type=float,
        help="Seconds without provider chunks before retrying; 0 disables (default)",
    )
    chat.add_argument("--max-steps", type=int)
    chat.add_argument("--max-calls", type=int)
    chat.add_argument("--tool-timeout", type=int)
    chat.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    chat.add_argument(
        "--reasoning-effort",
        choices=("default", "low", "medium", "high"),
        help="Provider reasoning effort; low may reduce latency but can affect quality",
    )
    chat.add_argument("--k", type=int)
    chat.add_argument("--mode", choices=("sequential", "parallel"))
    chat.add_argument("--test-command")
    chat.add_argument("--mcp-config", type=Path)
    continuation = sub.add_parser(
        "continue", help="Continue an interrupted coding run using its saved task and files"
    )
    continuation.add_argument("path", type=Path)
    continuation.add_argument("--max-steps", type=int, default=0)
    continuation.add_argument("--max-calls", type=int, default=0)
    continuation.add_argument("--timeout", type=int, default=0)
    continuation.add_argument("--tool-timeout", type=int)
    continuation.add_argument(
        "--max-tokens", type=int, help="Override output-token budget per response"
    )
    continuation.add_argument("--full-trace", action="store_true")
    replay = sub.add_parser("replay", help="Replay saved CLI events without model calls")
    replay.add_argument("path", type=Path, help="A run folder or trace.jsonl")
    replay.add_argument("--full-trace", action="store_true")
    replay.add_argument("--run", help="Only replay a particular run label, e.g. base1 or inv1.a1")
    return parser


def replay(args):
    paths = [args.path] if args.path.is_file() else sorted(args.path.rglob("trace.jsonl"))
    if not paths:
        raise ValueError("No trace.jsonl files found")
    events = []
    for path in paths:
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if not args.run or event["run"] == args.run:
                events.append(event)
    for event in sorted(events, key=lambda item: item["time"]):
        show_event(event, full=args.full_trace)
    return 0


def continue_run(args):
    root = args.path.expanduser().resolve()
    previous = json.loads((root / "config.json").read_text())
    if previous.get("mcp_enabled"):
        raise ValueError(
            "Continuing MCP runs is not supported yet; use a fresh run with your MCP configuration"
        )
    cwd = Path(previous.get("cwd", config.ROOT))
    repo = Path(previous["repo"])
    if not repo.is_absolute():
        repo = (cwd / repo).resolve()
    options = [
        previous["action"],
        "--repo",
        str(repo),
        "--task",
        previous["task"],
        "--model",
        previous["model"],
        "--output",
        str(root),
        "--resume",
        "--max-steps",
        str(args.max_steps),
        "--max-calls",
        str(args.max_calls),
        "--timeout",
        str(args.timeout),
    ]
    for name in (
        "harness",
        "test_command",
        "eval_command",
        "eval_tests",
        "check_timeout",
        "k",
        "mode",
        "library",
    ):
        if previous.get(name) is not None:
            value = previous[name]
            if name in ("harness", "eval_tests", "library") and not Path(value).is_absolute():
                value = str((cwd / value).resolve())
            options += ["--" + name.replace("_", "-"), str(value)]
    options += [
        "--max-tokens",
        str(args.max_tokens if args.max_tokens is not None else previous.get("max_tokens", 6000)),
    ]
    options += ["--tool-timeout", str(args.tool_timeout or previous.get("tool_timeout", 60))]
    if args.full_trace or previous.get("full_trace"):
        options.append("--full-trace")
    return main(options)


def main(argv=None):
    reliable_output()
    config.load_dotenv()
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    try:
        if args.action == "chat":
            from .chat import main_chat

            for name in ("max_tokens", "max_steps", "max_calls", "stream_idle_timeout"):
                if getattr(args, name) is not None and getattr(args, name) < 0:
                    parser.error(f"--{name.replace('_', '-')} must be zero or positive")
            for name in ("k", "tool_timeout"):
                if getattr(args, name) is not None and getattr(args, name) < 1:
                    parser.error(f"--{name.replace('_', '-')} must be positive")
            if sum(bool(value) for value in (args.repo, args.from_run, args.session)) != 1:
                parser.error("Choose exactly one of --repo, --from-run, or --session")
            return main_chat(args)
        if args.action == "replay":
            return replay(args)
        if args.action == "continue":
            return continue_run(args)
        if not args.repo.is_dir():
            parser.error("--repo must be an existing project directory")
        for flag in ("max_steps", "max_calls", "timeout"):
            if getattr(args, flag) < 0:
                parser.error(f"--{flag.replace('_', '-')} must be zero (unlimited) or positive")
        for flag in ("max_tokens", "tool_timeout", "check_timeout"):
            if getattr(args, flag) < 1:
                parser.error(f"--{flag.replace('_', '-')} must be positive")
        if getattr(args, "k", 1) < 1:
            parser.error("--k must be positive")
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            parser.error("Set OPENROUTER_API_KEY in the repository .env file")
        task = args.task if args.task is not None else args.task_file.read_text()
        if not task.strip():
            parser.error("The task must not be empty")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        root = (args.output or Path("runs/coding") / stamp).expanduser().resolve()
        if root.exists() and not args.resume:
            parser.error("--output already exists; use a new directory or --resume")
        if args.resume and not (root / "config.json").is_file():
            parser.error("--resume requires an existing output with config.json")
        repo = args.repo.expanduser().resolve()
        if repo == root or repo.is_relative_to(root):
            parser.error("The output directory cannot contain the source repository")
        if args.eval_tests:
            args.eval_tests = args.eval_tests.expanduser().resolve()
            if not args.eval_tests.exists() or args.eval_tests.is_relative_to(repo):
                parser.error(
                    "--eval-tests must exist outside --repo, so they are excluded from trial context"
                )
            if not args.eval_command:
                parser.error(
                    "--eval-tests requires --eval-command, e.g. python -m unittest discover -s .h2-eval"
                )
        if args.harness and not (args.harness / "systemprompt.md").is_file():
            parser.error("--harness must contain systemprompt.md")
        mcp_config = json.loads(args.mcp_config.read_text()) if args.mcp_config else None
        if args.resume:
            previous = json.loads((root / "config.json").read_text())
            current = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            current["task"] = task
            for setting in (
                "repo",
                "task",
                "model",
                "harness",
                "action",
                "mode",
                "k",
                "test_command",
                "eval_command",
                "eval_tests",
            ):
                old = previous.get(setting)
                new = current.get(setting)
                if setting == "repo":
                    old_path = Path(old)
                    old = (
                        str((Path(previous.get("cwd", config.ROOT)) / old_path).resolve())
                        if not old_path.is_absolute()
                        else str(old_path.resolve())
                    )
                    new = str(repo)
                if old != new:
                    parser.error(
                        f"Cannot resume with changed {setting}; use a new output directory"
                    )
            if args.mcp_config:
                parser.error("MCP runs currently require a fresh output directory")
        root.mkdir(parents=True, exist_ok=args.resume)
        trace = Trace(root / "trace.jsonl", "harness2", full=args.full_trace, secrets=(key,))
        trace.emit("stage", f"{args.action.upper()} | model={args.model} | output={root}")
        trace.emit(
            "stage",
            "Copying project; original files will not be overwritten. Shell tools execute locally, not in an OS sandbox.",
        )
        if not args.resume:
            blocked = (root,)
            if root.is_relative_to(repo) and root.relative_to(repo).parts[0] == "runs":
                blocked = (repo / "runs",)
            snapshot(repo, root / "source", excluded_roots=blocked)
        else:
            trace.emit(
                "stage",
                "RESUME: keeping partial workspaces, conversation checkpoints, and completed stages",
            )
        configuration = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "mcp_config"
        }
        configuration["task"] = trace.clean(task)
        configuration["repo"] = str(repo)
        configuration["cwd"] = str(Path.cwd())
        configuration["mcp_enabled"] = bool(mcp_config)
        if args.eval_tests and not args.resume:
            if args.eval_tests.is_dir():
                snapshot(args.eval_tests, root / "evaluation-tests")
            else:
                (root / "evaluation-tests").mkdir()
                shutil.copyfile(args.eval_tests, root / "evaluation-tests" / args.eval_tests.name)
        (root / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
        budget = Budget(args.max_calls)

        def model():
            return make_model(args.model, key, args.max_tokens)

        adapter = CodingAdapter(
            root=root,
            source=root / "source",
            task=task,
            model=model,
            model_id=args.model,
            budget=budget,
            trace=trace,
            max_steps=args.max_steps,
            timeout=args.timeout,
            tool_timeout=args.tool_timeout,
            harness=(root / "base-harness") if args.resume else args.harness,
            mcp_config=mcp_config,
            resume=args.resume,
        )
        commands = {}
        if args.test_command:
            commands["project_tests"] = args.test_command
        if args.eval_command:
            commands["held_out"] = args.eval_command
        if not commands:
            trace.emit(
                "stage",
                "No quality check configured: traces will show behavior, but improvement will be unmeasured.",
            )
        if args.action == "run":
            try:
                adapter.execute(adapter.task_id, None, f"coding-lg-base/{adapter.task_id}/base1")
                result = {"status": "ok", "selections": {}, "single_agent": True}
            except Exception as exc:
                trace.emit("error", f"Single-agent execution failed: {type(exc).__name__}: {exc}")
                result = {
                    "status": "failed",
                    "selections": {},
                    "errors": [str(exc)],
                    "single_agent": True,
                }
        else:
            mode = mode_token(args.mode)
            trace.emit(
                "stage",
                f"HARNESS2 {args.mode}, k={args.k}: baseline → propose → probe → compose → final → learn",
            )
            if args.k == 1:
                trace.emit(
                    "stage",
                    "At k=1 Harness2 adopts the single composed final; selection is not a test-based win over baseline.",
                )
            lib = Library(
                root=(args.library.resolve() if args.library else root / "library") / mode,
                domain="coding",
                surface=SURFACE,
            )
            trace.emit(
                "stage", f"Improvement library: {len(lib.entries())} existing lessons; {lib.root}"
            )
            proposer = CodingProposer(adapter)
            engine = run_task(
                adapter.task_id,
                args.k,
                adapter=adapter,
                prop=proposer,
                selector_prop=proposer,
                runs_root=adapter.spec.runs_root,
                modes=(mode,),
                libraries={mode: lib},
            )
            result = engine.to_dict()
            trace.emit("stage", f"LIBRARY: {len(lib.entries())} retained entries at {lib.root}")
            for error in result.get("errors", []):
                trace.emit("error", error)
            for mode_result in result.get("modes", {}).values():
                for stage in mode_result.get("stages", []):
                    if stage.get("fell_back"):
                        trace.emit(
                            "error",
                            f"Harness proposal {stage.get('label')} fell back: {stage.get('error')}",
                        )
            for selection in result.get("selections", {}).values():
                chosen = selection.get("chosen_label")
                if chosen:
                    paths = list(
                        (root / "executions").glob(f"*/{adapter.task_id}/{chosen}/harness")
                    )
                    if paths:
                        shutil.copytree(paths[0], root / "adopted-harness", dirs_exist_ok=True)
                        (root / "selected-workspace.txt").write_text(
                            str(paths[0].parent / "workspace") + "\n"
                        )
                    trace.emit(
                        "stage",
                        f"ADOPTED {chosen}; model selection is separate from external correctness checks",
                    )
        report = build_report(
            adapter,
            result=result,
            commands=commands,
            timeout=args.check_timeout,
            usage={
                "calls": budget.calls,
                "input_tokens": budget.input_tokens,
                "output_tokens": budget.output_tokens,
            },
            eval_tests=root / "evaluation-tests" if args.eval_tests else None,
        )
        trace.emit("result", f"Report: {root / 'report.md'}")
        trace.emit("result", f"Replay: h2-code replay {root}")
        selected = next(
            (row for row in report["rows"] if row["label"] == report["selected_label"]), None
        )
        if args.action == "run":
            selected = next((row for row in report["rows"] if row["label"] == "base1"), None)
        failed_checks = selected and any(
            not check["passed"] for check in selected["checks"].values()
        )
        return int(result["status"] not in ("ok", "skipped") or bool(failed_checks))
    except KeyboardInterrupt:
        CONSOLE.print("Interrupted. Completed traces and workspaces remain in the run folder.")
        return 130
    except Exception as exc:
        CONSOLE.print(f"Error: {type(exc).__name__}: {exc}", style="red", markup=False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
