"""Deterministic, hidden, five-switch experimental environment.

The model sees only public_task(), its probes and their observed results.
It never receives generated parameters or the implementation of signal().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import random

N_SWITCHES = 5
MODULUS = 7
ALL_SETTINGS = tuple(''.join(map(str, bits)) for bits in product((0, 1), repeat=N_SWITCHES))


def validate_setting(value: str) -> str:
    if not isinstance(value, str) or len(value) != N_SWITCHES or any(c not in '01' for c in value):
        raise ValueError(f'Setting must be exactly {N_SWITCHES} binary digits, e.g. 01001')
    return value


@dataclass(frozen=True)
class HiddenMachine:
    """Hidden data belongs exclusively to the environment, never the model prompt."""
    bias: int
    weights: tuple[int, ...]
    interaction: tuple[int, int, int] | None = None

    @classmethod
    def generate(cls, seed: int, family: str = 'additive') -> 'HiddenMachine':
        if family not in {'additive', 'interaction'}:
            raise ValueError('family must be additive or interaction')
        rng = random.Random(seed)
        bias = rng.randrange(MODULUS)
        weights = tuple(rng.randrange(1, MODULUS) for _ in range(N_SWITCHES))
        interaction = None
        if family == 'interaction':
            i, j = sorted(rng.sample(range(N_SWITCHES), 2))
            interaction = (i, j, rng.randrange(1, MODULUS))
        return cls(bias, weights, interaction)

    def signal(self, setting: str) -> int:
        setting = validate_setting(setting)
        result = self.bias + sum(int(bit) * weight for bit, weight in zip(setting, self.weights))
        if self.interaction is not None:
            i, j, effect = self.interaction
            result += int(setting[i]) * int(setting[j]) * effect
        return result % MODULUS


@dataclass
class Experiment:
    machine: HiddenMachine
    budget: int = 7
    observations: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.budget < 1 or self.budget >= len(ALL_SETTINGS):
            raise ValueError('budget must be between 1 and 31')

    def probe(self, setting: str) -> int:
        validate_setting(setting)
        if len(self.observations) >= self.budget:
            raise RuntimeError('Probe budget exhausted')
        result = self.machine.signal(setting)
        self.observations.append({'setting': setting, 'signal': result})
        return result

    def observed_settings(self) -> set[str]:
        return {item['setting'] for item in self.observations}

    def public_task(self) -> str:
        return (
            f'You are investigating an unfamiliar deterministic machine with {N_SWITCHES} binary switches. '
            f'A setting is a {N_SWITCHES}-character string of 0s and 1s, e.g. 01001. '
            f'The machine returns one integer signal from 0 to {MODULUS - 1}. '
            'You do not know how its output is generated. '
            'You may experimentally choose switch settings, observe their outputs, '
            'and ultimately predict the signal for all 32 possible settings. '
            'Your objective is accurate predictions for switch settings you have not tested. '
            'No particular scientific or mathematical method is required.'
        )

    def accuracy(self, predictions: dict[str, int]) -> dict:
        """Evaluate ONLY as-yet-unprobed settings. Evaluation has no tool feedback."""
        heldout = sorted(set(ALL_SETTINGS) - self.observed_settings())
        valid = (
            isinstance(predictions, dict)
            and set(predictions.keys()) == set(ALL_SETTINGS)
            and all(type(v) is int and 0 <= v < MODULUS for v in predictions.values())
        )
        correct = sum(predictions.get(s) == self.machine.signal(s) for s in heldout) if valid else 0
        return {
            'correct': correct, 'tested': len(heldout), 'accuracy': correct / len(heldout),
            'valid_prediction_map': valid,
            'random_guess_expectation': 1 / MODULUS,
        }
