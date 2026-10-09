# Autonomous Learning Lab — Experiment 0 (v0.1)

**Purpose:** Before engineering persistent memory, a Discoverer, or a Challenger, observe whether a plain pretrained reasoning model can infer an unfamiliar deterministic mechanism from controlled interaction. This is **within-episode adaptation**, not a demonstration of continual learning, meta-learning, a novel algorithm, or commercial value.

## What is the environment?

An unfamiliar machine takes **five binary switch settings** (`00000` to `11111`) and returns an integer signal **0–6**. The agent can choose **seven experiments**, then must predict the signal for all **32 settings**. Only **unprobed settings** count toward its score.

The machine is deterministic. Its hidden rule is generated from a seed. There are two families:

- `additive`: initially manageable with a few carefully selected experiments;
- `interaction`: introduces an additional hidden interaction that can invalidate a simple hypothesis.

**The model is never told the machine's formula, family, seed, or source.** Its prompts include only the public task description and its observations. No special prompts instruct the model to use a scientific method.

This is *not* meant to establish that LLMs uniquely enable learning. The additive family is solvable perfectly by a short classical algorithm that knows the form. That classical method is explicitly included as a ceiling/sanity check.

## Requirements

Python 3.10+, no pip packages. Windows PowerShell supported. Optional: a working authenticated `claude` command for the actual LLM experiment (Claude Max typically provides access subject to subscription limits).

## 1. Validate the lab **without** an LLM

From this directory in PowerShell:

```powershell
py -m unittest discover -s tests -v
py -m unknown_lab.demo
```

These verify that the simulator and scoring work. They do **not** demonstrate autonomous intelligence.

## 2. Run the preregistered first LLM observation

Outside an existing interactive Claude Code session, in normal PowerShell:

```powershell
claude --version
py -m unknown_lab.run --policy claude --model haiku --family additive --seed 1319 --budget 7 --out results/pilot-haiku-1319.json
```

The harness invokes `claude -p` with **no Claude tools** and a **fresh session for every decision**. Claude only receives the current task and observed trial log; it cannot inspect environment code or hidden parameters. There are nine model calls with default settings: one blind prediction, seven probe choices, and one final prediction.

Results include: initial accuracy, accuracy on *the same subsequently unprobed settings* after seven observations, observed probes, all predictions, and available Claude Code usage metadata. They do not include private reasoning traces. The harness does not rely on arbitrary model-generated executable code.

If this fails due to Claude CLI argument version differences, inspect `claude --help` and update the adapter **without changing the evaluation protocol**. Record any changes.

## 3. Offline controls (same environment and seed)

```powershell
py -m unknown_lab.run --policy memorize --family additive --seed 1319 --budget 7 --out results/memorize-1319.json
py -m unknown_lab.run --policy additive --family additive --seed 1319 --budget 7 --out results/classical-1319.json
```

`memorize` uses trials and remembers raw observations but makes **no structural inference**; it is deliberately weak. `additive` knows the hypothesis family in advance but not its parameters; it is a **privileged classical ceiling, not a matched baseline**. Neither control establishes anything about LLM advantage.

## 4. What next (NOT implemented)

If there is observable within-episode learning, run the same protocol on multiple seeds. Then test an unfamiliar interaction and compare an ordinary transcript with an autonomously written reusable procedure after a fresh context reset. Add a Challenger **only after** we measure a persistence/transfer effect and can isolate adaptive-challenge value under matched budgets.

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for interpretation, stop conditions, and threats to validity.
