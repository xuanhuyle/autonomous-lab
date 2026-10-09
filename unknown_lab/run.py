"""Run the minimal (no persistence) black-box investigation pilot.

Example: py -m unknown_lab.run --policy claude --model haiku --seed 1319
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import AdditiveReference, ClaudePolicy, MemorizeOnlyReference
from .engine import ALL_SETTINGS, Experiment, HiddenMachine, validate_setting


def predictions_prompt(task: str, observations: list[dict], initial: bool = False, notes: list[str] | None = None) -> str:
    detail = ('You have not conducted any experiments yet. Make your best predictions without probing.'
              if initial else 'No more probing is permitted. Make your best predictions using the observations below.')
    return (
        task + '\n\n' + detail + '\n'
        + 'Observations: ' + json.dumps(observations, separators=(',', ':')) + '\n'
        + 'Your own optional notes from previous decisions: ' + json.dumps(notes or []) + '\n'
        + 'Return ONLY one JSON object with a `predictions` field mapping EACH of the 32 '
          'five-digit settings to its predicted integer signal (0..6). '
          'Example structure: {"predictions":{"00000":0,"00001":1,...}}. '
          'Include all 32 keys; ellipses are not allowed.\n'
    )


def probe_prompt(task: str, observations: list[dict], notes: list[str], remaining: int) -> str:
    return (
        task + '\n\n'
        + f'You have {remaining} experimental probes remaining.\n'
        + 'Previous observations (chronological): '
        + json.dumps(observations, separators=(',', ':')) + '\n'
        + 'Your own optional notes from previous decisions: ' + json.dumps(notes[-7:]) + '\n'
        + 'Choose ONE setting to test next. You are free to use any reasoning or experimental strategy. '
          'You MAY include a short note to your future self about your current hypotheses. '
          'Return ONLY JSON, e.g. {"setting":"01001","note":"optional hypothesis"}. No commentary.\n'
    )


def simulate(policy_name: str, model: str, family: str, seed: int, budget: int, pretest: bool, verbose: bool = True) -> dict:
    experiment = Experiment(HiddenMachine.generate(seed, family), budget)
    task = experiment.public_task()
    policy = {
        'additive': lambda: AdditiveReference(),
        'memorize': lambda: MemorizeOnlyReference(seed),
        'claude': lambda: ClaudePolicy(model),
    }[policy_name]()
    initial_predictions = None
    if pretest:
        if policy_name == 'claude':
            response = policy.ask(predictions_prompt(task, [], initial=True), response_kind='predictions')
            initial_predictions = response.get('predictions', {})
        else:
            initial_predictions = policy.predictions([])
    decisions = []
    notes = []
    for turn in range(budget):
        if policy_name == 'claude':
            response = policy.ask(probe_prompt(task, experiment.observations, notes, budget - turn), response_kind='probe')
            setting = response.get('setting')
        else:
            setting = policy.next_probe(experiment.observations)
        try:
            validate_setting(setting)
        except ValueError as exc:
            decisions.append({'turn': turn + 1, 'error': str(exc), 'response': response if policy_name == 'claude' else None})
            break
        result = experiment.probe(setting)
        note = response.get('note') if policy_name == 'claude' else None
        if isinstance(note, str) and note.strip():
            notes.append(note[:500])
        decisions.append({'turn': turn + 1, 'setting': setting, 'observed_signal': result,
                          'model_note': note[:500] if isinstance(note, str) else None})
        if verbose:
            print(f'  probe {turn+1}/{budget}: {setting} -> {result}', flush=True)
    if policy_name == 'claude':
        response = policy.ask(predictions_prompt(task, experiment.observations, notes=notes), response_kind='predictions')
        final_predictions = response.get('predictions', {})
    else:
        final_predictions = policy.predictions(experiment.observations)
    post = experiment.accuracy(final_predictions)
    result = {
        'version': '0.1', 'policy': policy_name, 'model': model if policy_name == 'claude' else None,
        'family': family, 'seed': seed, 'probe_budget': budget, 'actual_probes': len(experiment.observations),
        'task_text': task, 'pretest': experiment.accuracy(initial_predictions) if initial_predictions is not None else None,
        'posttest': post, 'observations': experiment.observations, 'decisions': decisions,
        'final_predictions': final_predictions,
        'reference_model_has_privileged_family_assumption': policy_name == 'additive',
        'claude_usage': policy.usage if policy_name == 'claude' else [],
        'interpretation': 'Within-episode adaptation only. No persistent skill, meta-learning, or novel method proven.',
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description='Unknown machine learning pilot, no external dependencies')
    p.add_argument('--policy', choices=['memorize', 'additive', 'claude'], default='memorize')
    p.add_argument('--model', default='haiku', help='Claude Code model alias (only for --policy claude)')
    p.add_argument('--family', choices=['additive', 'interaction'], default='additive')
    p.add_argument('--seed', type=int, default=1319)
    p.add_argument('--budget', type=int, default=7)
    p.add_argument('--no-pretest', action='store_true', help='Skip initial model prediction to save one call')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    result = simulate(args.policy, args.model, args.family, args.seed, args.budget, not args.no_pretest)
    print(f"posttest held-out: {result['posttest']['correct']}/{result['posttest']['tested']} "
          f"({result['posttest']['accuracy']:.1%}); prediction map valid={result['posttest']['valid_prediction_map']}")
    if result['pretest']:
        print(f"pretest (same held-out states, before observations): {result['pretest']['accuracy']:.1%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('wrote', args.out)


if __name__ == '__main__':
    main()
