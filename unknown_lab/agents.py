"""Reference policies and a tool-less Claude Code model adapter."""
from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
from collections import Counter

from .engine import ALL_SETTINGS, MODULUS


def parse_json_answer(text: str) -> dict:
    """Parse a standalone JSON object, tolerating markdown code fences."""
    if not isinstance(text, str):
        raise ValueError('Expected model output text')
    content = text.strip()
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*', '', content, count=1, flags=re.I)
        content = re.sub(r'\s*```\s*$', '', content, count=1)
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError('Model response must be a JSON object')
    return result


def parse_claude_stream(stdout: str) -> tuple[dict, dict]:
    """Parse Claude CLI stream-json and recover text from assistant events.

    Some Claude Code releases emit a successful final result with an empty
    `result` field even when assistant messages contain the generated text.
    Parsing these messages avoids silently losing valid model responses.
    """
    events = []
    for i, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Claude emitted invalid stream JSON on line {i}') from exc
        if isinstance(obj, dict):
            events.append(obj)

    result = next((e for e in reversed(events) if e.get('type') == 'result'), None)
    if result is None:
        kinds = [e.get('type') for e in events[-5:]]
        raise RuntimeError(f'Claude produced no final result event (last event types: {kinds})')

    if result.get('is_error') or result.get('subtype') != 'success':
        raise RuntimeError(
            'Claude Code did not complete successfully: '
            f"subtype={result.get('subtype')!r}; "
            f"errors={result.get('errors')!r}; "
            f"message={str(result.get('result', ''))[:400]!r}"
        )

    candidates = []
    if isinstance(result.get('result'), str) and result['result'].strip():
        candidates.append(result['result'])

    for event in reversed(events):
        if event.get('type') != 'assistant':
            continue
        message = event.get('message') or {}
        blocks = message.get('content') if isinstance(message, dict) else None
        if not isinstance(blocks, list):
            continue
        message_text = ''.join(
            b['text'] for b in blocks
            if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str)
        )
        if message_text.strip():
            candidates.append(message_text)

    for candidate in candidates:
        try:
            return parse_json_answer(candidate), result
        except (json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError(
        'Claude finished but supplied no parseable JSON answer. '
        f"final_result_length={len(str(result.get('result') or ''))}; "
        f"assistant_messages={sum(e.get('type') == 'assistant' for e in events)}; "
        f"subtype={result.get('subtype')!r}. "
        'Run `claude --version` and the standalone diagnostic in the README.'
    )


# Response shapes are declared by the harness, not invented by the model.
# This constrains only output serialization: it does not disclose any hidden
# machine parameters or prescribe an experimental strategy.
def response_schema(kind: str) -> dict:
    if kind == 'probe':
        return {
            'type': 'object',
            'properties': {
                'setting': {'type': 'string', 'enum': list(ALL_SETTINGS)},
                'note': {'type': 'string'},
            },
            'required': ['setting'],
            'additionalProperties': False,
        }
    if kind == 'predictions':
        return {
            'type': 'object',
            'properties': {
                'predictions': {
                    'type': 'object',
                    'properties': {setting: {'type': 'integer', 'minimum': 0, 'maximum': 6}
                                   for setting in ALL_SETTINGS},
                    'required': list(ALL_SETTINGS),
                    'additionalProperties': False,
                },
            },
            'required': ['predictions'],
            'additionalProperties': False,
        }
    raise ValueError(f'Unknown response kind: {kind}')


def parse_claude_structured(stdout: str, kind: str) -> tuple[dict, dict]:
    """Read structured_output from Claude Code's documented --json-schema JSON envelope."""
    try:
        envelope = json.loads(stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Claude CLI returned invalid JSON envelope; inspect raw CLI output') from exc
    if not isinstance(envelope, dict):
        raise RuntimeError('Claude CLI returned non-object JSON envelope')
    if envelope.get('is_error') or envelope.get('subtype') != 'success':
        raise RuntimeError(
            'Claude CLI failed: '
            f"subtype={envelope.get('subtype')!r}; "
            f"errors={envelope.get('errors')!r}; "
            f"result_excerpt={str(envelope.get('result') or '')[:600]!r}"
        )
    answer = envelope.get('structured_output')
    if not isinstance(answer, dict):
        raise RuntimeError(
            'Claude CLI returned success but no structured_output. '
            f"result_excerpt={str(envelope.get('result') or '')[:800]!r}"
        )
    if kind == 'probe':
        if set(answer) - {'setting', 'note'} or answer.get('setting') not in ALL_SETTINGS or (
            'note' in answer and not isinstance(answer['note'], str)
        ):
            raise RuntimeError(f'Invalid probe response shape: {str(answer)[:400]!r}')
    elif kind == 'predictions':
        preds = answer.get('predictions')
        if not isinstance(preds, dict) or set(preds) != set(ALL_SETTINGS) or any(
            type(x) is not int or not 0 <= x <= 6 for x in preds.values()
        ) or set(answer) != {'predictions'}:
            raise RuntimeError('Invalid predictions response shape (expected 32 integer values 0..6)')
    else:
        raise ValueError(f'Unknown response kind: {kind}')
    return answer, envelope


class ClaudePolicy:
    """Every call is a fresh, tool-less CLI session; no secret files are in cwd."""
    def __init__(self, model: str = 'haiku', timeout: int = 180):
        self.model = model
        self.timeout = timeout
        self.usage: list[dict] = []

    def ask(self, prompt: str, response_kind: str) -> dict:
        cmd = [
            'claude', '-p', '--model', self.model, '--output-format', 'json',
            '--json-schema', json.dumps(response_schema(response_kind), separators=(',', ':')),
            '--tools', '', '--disallowedTools', 'mcp__*', '--no-session-persistence',
        ]
        with tempfile.TemporaryDirectory(prefix='blackbox-agent-') as cwd:
            try:
                process = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                                         cwd=cwd, timeout=self.timeout, encoding='utf-8')
            except FileNotFoundError as exc:
                raise RuntimeError('Claude Code not found on PATH; run `claude --version`') from exc
        if process.returncode != 0:
            raise RuntimeError(
                f'Claude Code exited {process.returncode}: stderr={process.stderr[-1000:]!r}; '
                f'stdout_tail={process.stdout[-400:]!r}'
            )
        answer, envelope = parse_claude_structured(process.stdout, response_kind)
        self.usage.append({k: envelope.get(k) for k in ('duration_ms', 'total_cost_usd', 'usage', 'modelUsage') if k in envelope})
        return answer


class AdditiveReference:
    """A PRIVILEGED classical ceiling: knows the family form, not the parameters."""
    def __init__(self):
        self.probe_order = ['00000'] + [s for s in ('10000', '01000', '00100', '00010', '00001')] + ['11000'] + [s for s in ALL_SETTINGS if s not in {'00000', '10000', '01000', '00100', '00010', '00001', '11000'}]

    def next_probe(self, observations: list[dict]) -> str:
        tested = {o['setting'] for o in observations}
        return next(s for s in self.probe_order if s not in tested)

    def predictions(self, observations: list[dict]) -> dict[str, int]:
        seen = {o['setting']: o['signal'] for o in observations}
        bias = seen.get('00000', 0)
        weights = [(seen.get(s, bias) - bias) % MODULUS for s in ('10000', '01000', '00100', '00010', '00001')]
        return {s: (bias + sum(int(b) * w for b, w in zip(s, weights))) % MODULUS for s in ALL_SETTINGS}


class MemorizeOnlyReference:
    """Weak sanity-check control: stores trials but does not infer a general rule."""
    def __init__(self, seed: int):
        self.rng = random.Random(seed + 100_000)

    def next_probe(self, observations: list[dict]) -> str:
        tested = {o['setting'] for o in observations}
        return self.rng.choice([s for s in ALL_SETTINGS if s not in tested])

    def predictions(self, observations: list[dict]) -> dict[str, int]:
        seen = {o['setting']: o['signal'] for o in observations}
        majority = Counter(seen.values()).most_common(1)[0][0] if seen else 0
        return {s: seen.get(s, majority) for s in ALL_SETTINGS}
