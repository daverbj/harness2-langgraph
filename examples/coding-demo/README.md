# Coding-agent demo

This dependency-free project deliberately has two failing tests: `clamp()` does
not enforce the upper bound or reject reversed bounds. Five tests describe the
expected behavior. From this directory, the initial check is:

```bash
python -m unittest discover -s tests -v
```

To let the coding agent repair a copy, run from the harness2-langgraph checkout:

```bash
h2-code chat --repo examples/coding-demo --output runs/my-first-chat \
  --test-command "python -m unittest discover -s tests -v"
```

Ask: `Fix clamp() so all existing tests pass. Run the tests and summarize the changes.`
Then use `/diff`, `/test`, and optionally `/apply`. The original example stays
unchanged until you apply the workspace's changes.

For a Harness² experiment from the same initial example, start another session
and use `/improve "Fix clamp() to satisfy its existing tests."` as the first
request. This distinguishes ordinary coding from an explicit refinement run.
