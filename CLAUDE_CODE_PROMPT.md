# Prompt for an independent Claude Code audit (after local preflight)

Read README.md, EXPERIMENT_PROTOCOL.md, unknown_lab/*.py, tests/*.py, and the JSON outputs in results/.

You are auditing a small research instrument, **not** helping sell a self-improving-agent thesis.

1. Check for hidden-rule leakage into model prompts, unintended tool access, cross-run contamination, and score leakage through traces or CLI state.
2. Verify that pretest and posttest use the exact same *unprobed* states and that invalid predictions do not inflate scores.
3. Identify weaknesses in the causal inference from a single run. Critique any claims of learning that actually show only parameter identification, retrieval, or in-context adaptation.
4. Check if an inexpensive conventional algorithm already solves the task. In particular, distinguish the additive-reference *privileged hypothesis class* from a fair baseline.
5. If you find bugs, fix only instrumentation defects, add tests, and document protocol-affecting changes. Do not add a Challenger, memory framework, skill engine, LLM routing, or enterprise features.
6. Write a short AUDIT.md with: observed vs inferred vs not demonstrated, exact reproducible commands, first-pilot metrics (if present), and GO/NO-GO to the next empirical gate.

Avoid passing this audit prompt into the black-box `claude -p` experiment itself; it exposes source assumptions. Do not edit evaluation code after seeing a result without versioning and new held-out seeds.
