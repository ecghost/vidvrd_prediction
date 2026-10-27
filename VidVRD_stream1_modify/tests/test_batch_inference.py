import unittest
import json
import pickle
import tempfile
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from reference_serial import StreamPredictor as SerialPredictor
from stream import StreamPredictor
from test_streaming import features, metadata, model
from dataset import index_videos, load_video_events
from replay import scheduled_batches
from utils.video_relation_detection_openvoc import eval_detection_scores


def event(video, pair, begin, data):
    return dict(video_id=video, pair_id=pair, features=data, metadata=metadata(begin, begin + 30))


class BatchInferenceTests(unittest.TestCase):
    def run_schedule(self, streams, batch_size, active_count):
        engine = StreamPredictor(model(), 2)
        outputs, finals = {}, {}
        calls = []
        hook = engine.model.featEmbedding.spatial_decoder.register_forward_hook(
            lambda module, inputs, output: calls.append(len(output[0])))
        with patch('replay.load_video_events', side_effect=lambda pairs, clip_len: streams[pairs]):
            for batch, finished in scheduled_batches([(video, video) for video in streams],
                                                      inference_batch_size=batch_size, max_active_videos=active_count):
                self.assertLessEqual(len(batch), batch_size)
                self.assertEqual(len({(item['video_id'], item['pair_id']) for item in batch}), len(batch))
                for result in engine.stream_step_batch(batch):
                    outputs.setdefault(result['video_id'], []).append(result)
                for video, clock in finished:
                    if clock is not None:
                        ending = engine.finish(video, clock)
                        for entry in ending:
                            self.assertGreaterEqual(entry['emitted_at_fid'], entry['end_fid'])
                    finals[video] = engine.association.final(video)
                    engine.reset(video)
        hook.remove()
        self.assertFalse(engine.caches)
        return outputs, finals, calls

    def test_scheduler_sizes_order_prefix_tails_empty_and_final_format(self):
        data = features(6)
        current = lambda index: {key: value[0, index] for key, value in data.items()}
        streams = {'v1': [], 'v2': [], 'empty': [], 'v3': []}
        for video, count in [('v1', 6), ('v2', 3), ('v3', 1)]:
            for index in range(count):
                begin = index * 30 + (30 if index >= 3 else 0)
                for pair in ('p', 'q', 'r'):
                    if pair == 'r' and index % 2:
                        continue
                    streams[video].append((begin + 30, pair, current(index), metadata(begin, begin + 30)))
        reference = SerialPredictor(model(), 2)
        expected, expected_finals = {}, {}
        for video, events in streams.items():
            expected[video] = [reference.stream_step(video, pair, feature, meta, clock)
                               for clock, pair, feature, meta in events]
            if events:
                reference.finish(video, events[-1][0])
            expected_finals[video] = reference.association.final(video)
            reference.reset(video)
        expected.pop('empty')
        for batch_size, active in [(1, 1), (2, 1), (4, 2), (7, 3)]:
            actual, finals, calls = self.run_schedule(streams, batch_size, active)
            self.assertEqual(set(actual), set(expected))
            for video in actual:
                self.assertEqual(len(actual[video]), len(expected[video]))
                for old, new in zip(expected[video], actual[video]):
                    torch.testing.assert_close(torch.tensor(old['scores']), torch.tensor(new['scores']), atol=2e-5, rtol=2e-5)
                    for left, right in zip(old['events'], new['events']):
                        self.assertEqual({key: value for key, value in left.items() if key not in ('score', 'score_sum', 'predicate_score')},
                                         {key: value for key, value in right.items() if key not in ('score', 'score_sum', 'predicate_score')})
                    self.assertEqual(len(old['events']), len(new['events']))
                order = lambda item: (item['triplet'], item['duration'])
                for old, new in zip(sorted(expected_finals[video], key=order), sorted(finals[video], key=order)):
                    self.assertEqual({key: value for key, value in old.items() if key != 'score'},
                                     {key: value for key, value in new.items() if key != 'score'})
                    self.assertAlmostEqual(old['score'], new['score'], places=5)
                    self.assertEqual(len(new['sub_traj']), new['duration'][1] - new['duration'][0])
                json.dumps(finals[video])
                eval_detection_scores(finals[video], finals[video], .5)
            self.assertEqual(sum(calls), sum(len(events) for events in streams.values()))
            if batch_size > 1:
                self.assertGreater(max(calls), 1)
            prefix_streams = {'v1': streams['v1'][:8]}
            prefix, _, _ = self.run_schedule(prefix_streams, batch_size, active)
            for old, new in zip(actual['v1'][:8], prefix['v1']):
                torch.testing.assert_close(torch.tensor(old['scores']), torch.tensor(new['scores']), atol=2e-5, rtol=2e-5)
                self.assertEqual([(item['event'], item['predicate_id'], item['end_fid']) for item in old['events']],
                                 [(item['event'], item['predicate_id'], item['end_fid']) for item in new['events']])
        self.assertEqual(self.run_schedule({}, 4, 2)[:2], ({}, {}))

    def test_lazy_index_and_active_video_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {key: value[0].numpy() for key, value in features(1, 768).items()}
            for video in ('a', 'b', 'c'):
                meta = metadata(0, 30)
                meta['duration'] = [0, 30]
                with (root / f'{video}_000000.pkl').open('wb') as handle:
                    pickle.dump([arrays, meta], handle)
            with patch('dataset.read_pair', side_effect=AssertionError('index must not read features')):
                indexed = index_videos(root, max_videos=2)
            self.assertEqual([video for video, _ in indexed], ['a', 'b'])
            loaded = []
            def load(pairs, clip_len):
                loaded.append(pairs[0][1].stem)
                return load_video_events(pairs, clip_len)
            with patch('replay.load_video_events', side_effect=load):
                schedule = scheduled_batches(index_videos(root), inference_batch_size=1, max_active_videos=2)
                first_batch, _ = next(schedule)
                self.assertEqual(len(loaded), 2)
                self.assertEqual(first_batch[0]['video_id'], 'a')
                list(schedule)
                self.assertEqual(len(loaded), 3)

    def test_completed_video_input_released_before_replacement_load(self):
        references = []
        def load(pairs, clip_len):
            if references:
                self.assertIsNone(references[-1]())
            data = {key: value[0, 0] for key, value in features(1).items()}
            references.append(weakref.ref(data['clip_feat']))
            return [(30, 'p', data, metadata(0, 30))]
        with patch('replay.load_video_events', side_effect=load):
            schedule = scheduled_batches([('a', 'a'), ('b', 'b')], max_active_videos=1)
            batch, finished = next(schedule)
            self.assertEqual(finished, [('a', 30)])
            del batch
            batch, finished = next(schedule)
            self.assertEqual(finished, [('b', 30)])
            del batch
            self.assertEqual(list(schedule), [])
            self.assertIsNone(references[-1]())

    def test_ties_keep_numpy_selection_rule_and_single_batch_encoding(self):
        net = model()
        with torch.no_grad():
            net.text_subject[:] = net.text_subject[0].clone()
            net.text_object[:] = net.text_object[0].clone()
            net.text_predicate[:] = net.text_predicate[0].clone()
        serial, engine = SerialPredictor(net, 2), StreamPredictor(net, 2)
        data = {key: value[0, 0] for key, value in features(1).items()}
        items = [event('v', pair, 0, data) for pair in ('a', 'b', 'c')]
        expected = [serial.stream_step(**item) for item in items]
        sizes = []
        hook = net.featEmbedding.spatial_decoder.register_forward_hook(lambda module, inputs, result: sizes.append(len(result[0])))
        actual = engine.stream_step_batch(items)
        hook.remove()
        self.assertEqual(sizes, [3])
        for old, new in zip(expected, actual):
            self.assertEqual([entry['predicate_id'] for entry in old['events']], [entry['predicate_id'] for entry in new['events']])

    def test_batch_matches_frozen_serial_with_variable_windows_and_sliding(self):
        net = model()
        serial, batched = SerialPredictor(net, 2), StreamPredictor(net, 2)
        data = features(6)
        for index in range(6):
            current = {key: value[0, index] for key, value in data.items()}
            events = [event('v1', 'p', index * 30, current), event('v2', 'p', index * 30, current)]
            if index % 2 == 0:
                events.append(event('v1', 'q', index * 30, current))
            expected = [serial.stream_step(**item) for item in events]
            actual = batched.stream_step_batch(events)
            for reference, result in zip(expected, actual):
                torch.testing.assert_close(torch.tensor(reference['scores']), torch.tensor(result['scores']), atol=2e-5, rtol=2e-5)
                self.assertEqual([(entry['event'], entry['predicate_id'], entry['start_fid'], entry['end_fid'])
                                  for entry in reference['events']],
                                 [(entry['event'], entry['predicate_id'], entry['start_fid'], entry['end_fid'])
                                  for entry in result['events']])
            for key in serial.caches:
                self.assertEqual(len(serial.caches[key]), len(batched.caches[key]))
                for old, new in zip(serial.caches[key], batched.caches[key]):
                    for old_tensor, new_tensor in zip(old, new):
                        torch.testing.assert_close(old_tensor, new_tensor, atol=2e-5, rtol=2e-5)
        batched.finish('v1', 180)
        self.assertNotIn(('v1', 'p'), batched.caches)
        self.assertIn(('v2', 'p'), batched.caches)
        batched.reset('v1')
        self.assertIn(('v2', 'p'), batched.caches)

    def test_reject_duplicate_pair_and_time_group_before_mutation(self):
        engine = StreamPredictor(model())
        current = {key: value[0, 0] for key, value in features(1).items()}
        for events in ([event('v', 'p', 0, current), event('v', 'p', 30, current)],
                       [event('v', 'p', 0, current), event('v', 'q', 30, current)]):
            with self.assertRaises(ValueError):
                engine.stream_step_batch(events)
            self.assertFalse(engine.caches or engine.clocks)
        self.assertEqual(engine.stream_step_batch([]), [])
