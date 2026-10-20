from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from rich.table import Table

from .files import command, snapshot
from .trace import CONSOLE


def evaluate(workspace, test_command, timeout, extra_tests=None, cancel_event=None):
    start = time.monotonic()
    # Evaluation runs in a fresh copy, so one check cannot change a later check's input.
    with tempfile.TemporaryDirectory(prefix="h2-evaluate-") as temporary:
        target = Path(temporary) / "project"
        snapshot(workspace, target)
        if extra_tests:
            tests = target / ".h2-eval"
            if extra_tests.is_dir():
                snapshot(extra_tests, tests)
            else:
                tests.mkdir()
                shutil.copyfile(extra_tests, tests / extra_tests.name)
        result = command(test_command, target, timeout, cancel_event=cancel_event)
    result["wall_s"] = round(time.monotonic() - start, 2)
    counts = {}
    for name in ("passed", "failed", "skipped", "errors"):
        pattern = r"(\d+) " + ("errors?" if name == "errors" else name) + r"\b"
        matches = re.findall(pattern, result["output"])
        if matches:
            counts[name] = int(matches[-1])
    unittest_count = re.search(r"Ran (\d+) tests?", result["output"])
    if unittest_count and result["exit_code"] == 0 and re.search(r"^OK\b", result["output"], re.M):
        counts["passed"] = int(unittest_count.group(1))
    if unittest_count and result["exit_code"] != 0:
        counts["total"] = int(unittest_count.group(1))
        for key in ("failures", "errors", "skipped"):
            match = re.search(key + r"=(\d+)", result["output"])
            if match:
                counts[key] = int(match.group(1))
    result["test_counts"] = counts
    no_tests = bool(unittest_count and int(unittest_count.group(1)) == 0)
    result["passed"] = result["exit_code"] == 0 and not result["timed_out"] and not no_tests
    if no_tests:
        result["validation_note"] = "No tests were collected"
    return result


def build_report(adapter, *, result, commands, timeout, usage, eval_tests=None):
    root = adapter.root
    trace = adapter.trace
    targets = [("unmodified source", adapter.source, None)]
    for path in sorted((root / "executions").rglob("answer.txt")):
        run = path.parent
        metrics = json.loads((run / "metrics.json").read_text())
        targets.append((metrics["label"], run / "workspace", run))
    rows = []
    for label, workspace, run in targets:
        row = {"label": label, "workspace": str(workspace), "checks": {}}
        if run:
            row.update(json.loads((run / "metrics.json").read_text()))
            row["patch"] = str(run / "changes.patch")
        for kind, cmd in commands.items():
            trace.emit(
                "stage", f"VALIDATE {label}: {kind} (external check, after refinement)", command=cmd
            )
            check = evaluate(
                workspace,
                cmd,
                timeout,
                extra_tests=eval_tests if kind == "held_out" else None,
                cancel_event=getattr(adapter, "cancel_event", None),
            )
            check["output"] = trace.clean(check["output"])
            row["checks"][kind] = check
            trace.emit(
                "quality",
                f"{label}: {kind} {'PASS' if check['passed'] else 'FAIL'}; "
                f"exit={check['exit_code']}; counts={check['test_counts'] or 'not reported'}",
                output=check["output"],
            )
        rows.append(row)
    selections = result.get("selections", {})
    selected = next(iter(selections.values()), {})
    chosen = selected.get("chosen_label")
    original = next((r for r in rows if r["label"].startswith("base")), None)
    final = next((r for r in rows if r["label"] == chosen), None)
    primary = "held_out" if "held_out" in commands else "project_tests"
    verdict = "Not measured: no validation command provided."
    if commands and original and final:
        base = original["checks"][primary]
        composed = final["checks"][primary]
        if not base["passed"] and composed["passed"]:
            if selected.get("chosen_tree_is_base"):
                verdict = (
                    f"Final passed {primary} after baseline failed, but the harness is unchanged. "
                    "This is a better repeated attempt, not evidence of harness improvement."
                )
            else:
                verdict = (
                    f"Improved on this task: baseline failed {primary}; selected final passed."
                )
        elif base["passed"] and not composed["passed"]:
            verdict = f"Regression on this task: baseline passed {primary}; selected final failed."
        else:
            b, f = base["test_counts"], composed["test_counts"]
            if not base["passed"] and not composed["passed"] and b and f and b != f:
                verdict = f"Mixed result on this task: baseline counts {b}; selected final counts {f}. Inspect failures."
            else:
                verdict = "No measured improvement: baseline and selected final have the same check outcome."
    elif commands:
        verdict = (
            "Single agent run: validation measured; no Harness2 comparison requested."
            if result.get("single_agent")
            else "No completed baseline/final pair: improvement could not be measured."
        )
    if result.get("status") not in ("ok", "skipped"):
        verdict += " Harness2 workflow is incomplete; inspect errors."
    report = {
        "verdict": verdict,
        "selected_label": chosen,
        "selected_harness_is_base": selected.get("chosen_tree_is_base"),
        "fallback_stages": [
            stage["label"]
            for mode in result.get("modes", {}).values()
            for stage in mode.get("stages", [])
            if stage.get("fell_back")
        ],
        "commands": commands,
        "rows": rows,
        "usage": usage,
        "engine_result": result,
        "limitations": [
            "Single task; this does not establish overall quality.",
            "A command returning zero is not proof of comprehensive correctness.",
            "Same-task trials use additional model calls; this is not an equal-budget comparison.",
            "Evaluation results were not supplied to the Harness2 improvement/selection process.",
        ],
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    table = Table(title="Coding quality: baseline agent versus refined agent")
    table.add_column("Execution")
    for kind in commands:
        table.add_column(kind)
    table.add_column("Agent time")
    for row in rows:
        values = [row["label"] + (" (selected)" if row["label"] == chosen else "")]
        for kind in commands:
            check = row["checks"][kind]
            values.append(
                ("PASS" if check["passed"] else "FAIL")
                + (
                    f" {check['test_counts']}"
                    if check["test_counts"]
                    else f" exit {check['exit_code']}"
                )
            )
        values.append(f"{row['wall_s']:.1f}s" if "wall_s" in row else "—")
        table.add_row(*values)
    CONSOLE.print(table)
    trace.emit("quality", verdict)
    trace.emit(
        "result",
        f"Model requests: {usage['calls']}; input tokens: {usage['input_tokens']}; "
        f"output tokens: {usage['output_tokens']} (provider usage, not a dollar estimate)",
    )
    lines = [
        "# Coding agent run",
        "",
        verdict,
        "",
        f"Selected execution: `{chosen}`",
        "",
        "| Execution | " + " | ".join(commands) + (" |" if commands else "|"),
        "|---|" + "---|" * len(commands),
    ]
    for row in rows:
        values = [row["label"]] + [
            ("PASS" if row["checks"][kind]["passed"] else "FAIL") for kind in commands
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## What this measures",
        "",
        *["- " + item for item in report["limitations"]],
        "",
        "Project patches, workspaces and traces are retained under executions/.",
        "Instruction diffs and notes are retained under improver/.",
        "The adopted instructions are exported to adopted-harness/ when selection succeeds.",
        "The source repository was not overwritten.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return report
