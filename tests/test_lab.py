import unittest

import json
from unittest.mock import patch, Mock

from unknown_lab.agents import (AdditiveReference, ClaudePolicy, parse_json_answer,
                                parse_claude_stream, parse_claude_structured, response_schema)
from unknown_lab.engine import ALL_SETTINGS, Experiment, HiddenMachine
from unknown_lab.run import simulate


class ExperimentTests(unittest.TestCase):
    def test_signal_bounded_and_reproducible(self):
        for family in ('additive', 'interaction'):
            a = HiddenMachine.generate(1319, family)
            b = HiddenMachine.generate(1319, family)
            self.assertEqual(a, b)
            self.assertEqual(len(ALL_SETTINGS), 32)
            self.assertTrue(all(0 <= a.signal(s) < 7 for s in ALL_SETTINGS))

    def test_invalid_setting_or_excess_budget_rejected(self):
        e = Experiment(HiddenMachine.generate(1), 1)
        with self.assertRaises(ValueError):
            e.probe('111')
        e.probe('00000')
        with self.assertRaises(RuntimeError):
            e.probe('11111')

    def test_heldout_does_not_include_probed(self):
        e = Experiment(HiddenMachine.generate(2), 7)
        for s in ALL_SETTINGS[:7]:
            e.probe(s)
        perfect = {s: e.machine.signal(s) for s in ALL_SETTINGS}
        self.assertEqual(e.accuracy(perfect)['tested'], 25)
        self.assertEqual(e.accuracy(perfect)['accuracy'], 1.0)

    def test_missing_or_invalid_predictions_score_zero(self):
        e = Experiment(HiddenMachine.generate(2))
        self.assertFalse(e.accuracy({'00000': 0})['valid_prediction_map'])
        self.assertEqual(e.accuracy({'00000': 0})['correct'], 0)

    def test_additive_ceiling_after_six_probes(self):
        for seed in range(20):
            e = Experiment(HiddenMachine.generate(seed, 'additive'), 7)
            agent = AdditiveReference()
            for _ in range(6):
                e.probe(agent.next_probe(e.observations))
            self.assertEqual(e.accuracy(agent.predictions(e.observations))['accuracy'], 1.0)

    def test_interaction_is_detectable_by_generalization_failure(self):
        e = Experiment(HiddenMachine.generate(5, 'interaction'), 7)
        agent = AdditiveReference()
        for _ in range(7):
            e.probe(agent.next_probe(e.observations))
        self.assertLess(e.accuracy(agent.predictions(e.observations))['accuracy'], 1.0)

    def test_parser(self):
        self.assertEqual(parse_json_answer('```json\n{"setting":"01001"}\n```')['setting'], '01001')
        with self.assertRaises(ValueError):
            parse_json_answer('not JSON')

    def test_claude_stream_recovers_response_when_final_result_empty(self):
        lines = [
            {'type': 'system', 'subtype': 'init'},
            {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': '{"setting":"01001"}'}]}},
            {'type': 'result', 'subtype': 'success', 'is_error': False, 'result': '', 'total_cost_usd': 0.01},
        ]
        answer, meta = parse_claude_stream('\n'.join(json.dumps(line) for line in lines))
        self.assertEqual(answer['setting'], '01001')
        self.assertEqual(meta['total_cost_usd'], 0.01)

    def test_claude_stream_surfaces_execution_errors(self):
        event = {'type': 'result', 'subtype': 'error_during_execution', 'is_error': True,
                 'result': '', 'errors': ['authentication required']}
        with self.assertRaisesRegex(RuntimeError, 'authentication required'):
            parse_claude_stream(json.dumps(event))

    def test_claude_stream_requires_parseable_model_answer(self):
        lines = [
            {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Not JSON'}]}},
            {'type': 'result', 'subtype': 'success', 'is_error': False, 'result': ''},
        ]
        with self.assertRaisesRegex(RuntimeError, 'no parseable JSON'):
            parse_claude_stream('\n'.join(json.dumps(line) for line in lines))

    def test_schema_requests_exactly_32_predictions(self):
        schema = response_schema('predictions')
        shape = schema['properties']['predictions']
        self.assertEqual(set(shape['required']), set(ALL_SETTINGS))
        self.assertFalse(shape['additionalProperties'])

    def test_structured_predictions_success(self):
        predictions = {x: 2 for x in ALL_SETTINGS}
        envelope = {'type': 'result', 'subtype': 'success', 'is_error': False,
                    'structured_output': {'predictions': predictions},
                    'result': 'anything', 'total_cost_usd': 0.004}
        data, meta = parse_claude_structured(json.dumps(envelope), 'predictions')
        self.assertEqual(data['predictions'], predictions)
        self.assertEqual(meta['total_cost_usd'], 0.004)

    def test_structured_probe_success(self):
        envelope = {'subtype': 'success', 'is_error': False,
                    'structured_output': {'setting': '00010', 'note': 'test'}}
        data, _ = parse_claude_structured(json.dumps(envelope), 'probe')
        self.assertEqual(data['setting'], '00010')

    def test_structured_rejects_missing_or_bad_fields(self):
        for response in ({'predictions': {'00000': 1}}, {'predictions': {x: '2' for x in ALL_SETTINGS}}):
            envelope = {'subtype': 'success', 'structured_output': response}
            with self.assertRaisesRegex(RuntimeError, 'Invalid predictions response'):
                parse_claude_structured(json.dumps(envelope), 'predictions')
        with self.assertRaisesRegex(RuntimeError, 'no structured_output'):
            parse_claude_structured(json.dumps({'subtype': 'success', 'result': 'plain text'}), 'probe')

    def test_claude_policy_uses_schema(self):
        envelope = {'subtype': 'success', 'structured_output': {'setting': '01001'},
                    'total_cost_usd': 0.002}
        with patch('unknown_lab.agents.subprocess.run', return_value=Mock(
            returncode=0, stdout=json.dumps(envelope), stderr=''
        )) as call:
            policy = ClaudePolicy('haiku')
            self.assertEqual(policy.ask('pick', response_kind='probe')['setting'], '01001')
            cmd = call.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--output-format') + 1], 'json')
            self.assertIn('--json-schema', cmd)
            self.assertEqual(policy.usage[0]['total_cost_usd'], 0.002)

    def test_sanity_pipeline(self):
        result = simulate('memorize', 'none', 'additive', 1319, 7, True, verbose=False)
        self.assertEqual(result['posttest']['tested'], 25)
        self.assertEqual(result['actual_probes'], 7)
        self.assertEqual(result['family'], 'additive')


if __name__ == '__main__':
    unittest.main()
