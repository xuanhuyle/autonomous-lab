# Instrument status — 2026-10-09

## Executed locally (no LLM)

- `python -m unittest discover -s tests -v`: **8/8 passed**.
- `python -m unknown_lab.demo` using seeds 1319–1323, seven probes per machine:

| Hidden family | Naive memory-only control | Privileged additive classical reference |
|---|---:|---:|
| Additive | 11.2% | 100.0% |
| Pair interaction | 10.4% | 68.0% |

These are **offline controls only**, not evidence of LLM learning. Reference knows that the target family might be additive and thus has structural prior information. The naive control is deliberately weak.

- On preregistered seed 1319 / additive: naive control scored 3/25 (12%) and privileged reference scored 25/25 (100%).
- On seed 1319 / interaction (diagnostic only): privileged additive reference scored 17/25 (68%).

## Not yet executed

The Claude Code LLM calls **have not been run** here. This environment does not have a Claude Code executable/authenticated session. The provided command should be run from a local PowerShell session where `claude --version` works.

## Next concrete action

Run the single preregistered pilot:

`py -m unknown_lab.run --policy claude --model haiku --family additive --seed 1319 --budget 7 --out results/pilot-haiku-1319.json`

Bring back `results/pilot-haiku-1319.json` for an evidence-based assessment before altering the experiment.
