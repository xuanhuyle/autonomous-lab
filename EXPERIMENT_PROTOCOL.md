# Experiment 0: Pre-registered pilot protocol

## Scope, hypothesis, and limits

**Narrow question:** Can a frozen reasoning model, starting with no observations of a randomly generated machine, use seven self-chosen experiments to make more accurate predictions of its untested behavior than it could before experimenting?

The first experiment does **not** test persistent learning across episodes, the scientific status of a knowledge library, a game-theoretic Challenger, model independence, or learning to learn. It exists to observe and record the basic empirical behavior before deciding what machinery is missing.

## Primary outcome

Accuracy on all 32 possible settings **excluding settings the agent probed**. Post-probe predictions are compared against the hidden deterministic machine; pre-probe predictions are scored on the **same held-out settings** selected by the agent's future probes.

Secondary metrics: validity of the JSON map, probe duplication, empirical use of informative tests, model calls/usage metadata, and optionally documented qualitative insights (not acceptance criteria).

A score is valid only if the agent outputs integer predictions 0–6 for all 32 configurations. Invalid output = 0 correct. No hidden rule, source code, or evaluation feedback is provided to the agent.

## Preregistered first pilot

- Device: five binary switches, deterministic output 0–6.
- Family: `additive`; hidden parameters generated from seed `1319`.
- Model: Claude Code `haiku` alias (model version resolves locally, record it).
- Probe budget: **7** settings.
- No tool access; new Claude session on every invocation; same within-episode observation history is replayed.
- Baselines: `memorize` (weak); `additive` (knows formula family, privileged upper bound).
- Read initial and final accuracy; don't change rule family or probes after looking at results and call it the same pilot.

The family `interaction` and seeds `1320..1323` are reserved for **later diagnostic runs**, not the first pilot's primary result. An exploratory run using them must be labeled as such.

## Interpretation

- If **posttest > pretest**: evidence of *within-episode adaptation to provided observations* on this device. Not proof of new procedural knowledge or model-independent capability.
- If **posttest = pretest**: no observed learning advantage; may be poor experimental decisions, insufficient output expressivity, or the model failing to infer the mechanism.
- If `additive` = 100%: the instrument works and the mechanism is efficiently identifiable by a conventional algorithm **given the right hypothesis class**. It is not an LLM win.
- If Haiku instantly produces the right functional form from its priors: the experiment is a known-method identification test, not discovery of a new algorithm.
- A single seed and one model run **cannot** establish reliability; multiple seeds and repeated runs are required before concluding robustness.

## Identification and leakage limitations

- A hidden rule is not necessarily unknown as a *concept*: the model might have encountered modular linear functions in pretraining. This experiment tests adaptation to *unknown parameters*, not absence of mathematical prior knowledge.
- The tester knows the generative family and a classical reference does too. The Claude prompt does **not** say the family is additive or interactive, although the benchmark's design still privileges certain mathematical priors.
- The model gets a new context at each decision, but the harness carries forward the observations. This is **in-context adaptation**, not external persistent memory.
- Exposing *all* prior experiments can allow simple pattern recognition; a gain does not prove a scientific learning algorithm.
- Prompt can be contaminated if tools are accidentally enabled; inspect adapter and CLI version before running.

## Future gates (conditional; don't pre-build)

1. **Persistence:** Across fresh problems and fresh contexts, does a self-maintained knowledge artifact beat ordinary retrieval of the same raw experience, with comparable information and cost?
2. **Revision:** When a rule changes, does the agent detect the contradiction, preserve observations, and amend or scope beliefs without a human accepting each discovery?
3. **Game theory:** Does an adaptive Challenger improve *held-out learning efficiency* more than random challenges or a strong fixed active-learning curriculum under equal experiment budgets? Compare with PAIRED / classical active learning where relevant.
4. **Transfer and economics:** Does cumulative experience accelerate learning on **new families**, survive model swaps, and reduce *total* amortized cost against robust alternatives?

Each gate requires separately specified test data and stopping conditions. Passing Gate 0 alone must **not** be represented as evidence of these stronger claims.
