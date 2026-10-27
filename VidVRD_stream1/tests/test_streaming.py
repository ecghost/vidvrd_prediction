import json
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset import WindowDataset, cached_events, padding_collate_fn
from model_zoo.stage1 import Model, TemporalDecoder, focal, load_checkpoint, valid_mean
from stream import StreamPredictor
from utils.post_process import OnlineAssociation
from utils.video_relation_detection_openvoc import eval_detection_scores, evaluate

torch.set_num_threads(1)


def split(count):
    names = {str(index): f'class{index}' for index in range(count)}
    return dict(id2cls=names, cls2id={name: int(index) for index, name in names.items()},
                cls2split={name: 'base' if index % 2 == 0 else 'novel' for index, name in enumerate(names.values())})


def model():
    torch.manual_seed(7)
    text = {key: torch.randn(3 if key == 'objects' else 4, 16)
            for key in ('subject', 'object', 'predicate', 'objects')}
    return Model(text, split(4), split(3), window_size=3, src_split='all', dim=16, heads=4).eval()


def features(length=6, dim=16):
    return {'clip_feat': torch.randn(1, length, 4, dim), 'bbox_feat': torch.randn(1, length, 4, 24),
            'rel_feat': torch.randn(1, length, 42), 'mot_feat': torch.randn(1, length, 42)}


def metadata(begin, end):
    return dict(begin_fid=begin, end_fid=end, sbj_cls='subject', obj_cls='object',
                sbj_scr=0.8, obj_scr=0.9, sbj_traj=[[0, 0, 2, 2]] * (end - begin),
                obj_traj=[[1, 1, 3, 3]] * (end - begin))


class StreamingTests(unittest.TestCase):
    def test_evaluation_empty_thresholded_predictions(self):
        truth = dict(triplet=['a', 'b', 'c'], duration=[0, 30],
                     sub_traj=[[0, 0, 2, 2]] * 30, obj_traj=[[0, 0, 2, 2]] * 30)
        mean_ap, recall, _ = evaluate({'v': [truth]}, {'v': []})
        self.assertEqual(mean_ap, 0.)
        self.assertEqual(recall, {50: 0., 100: 0.})

    def test_temporal_causal_padding_and_capacity(self):
        torch.manual_seed(3)
        temporal = TemporalDecoder(16, 4).eval()
        tokens = torch.randn(2, 4, 4, 16)
        lengths = torch.tensor([4, 2])
        with torch.no_grad():
            original, interaction = temporal(tokens, lengths)
            changed = tokens.clone()
            changed[0, 2:] += torch.randn_like(changed[0, 2:]) * 50
            changed[1, 2:] = 500
            updated, new_interaction = temporal(changed, lengths)
        torch.testing.assert_close(original[:, :2], updated[:, :2])
        torch.testing.assert_close(interaction[:, :2], new_interaction[:, :2])
        with self.assertRaises(ValueError):
            temporal(torch.randn(1, 41, 4, 16), torch.tensor([41]))

    def test_padding_means_and_current_selection(self):
        net = model()
        data = features(3)
        data = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in data.items()}
        lengths = torch.tensor([1, 3])
        with torch.no_grad():
            original = net(data, lengths)
            for key in data:
                data[key][0, 1:] = 1000
            updated = net(data, lengths)
            single = net({key: value[:1, :1] for key, value in data.items()}, torch.tensor([1]))
        torch.testing.assert_close(original, updated)
        torch.testing.assert_close(updated[:1], single, atol=1e-5, rtol=1e-5)
        values = torch.tensor([[[1.], [999.], [999.]], [[1.], [2.], [3.]]])
        torch.testing.assert_close(valid_mean(values, lengths), torch.tensor([[1.], [2.]]))

    def test_cache_window_prediction_isolation_reset(self):
        net = model()
        stream = StreamPredictor(net)
        data = features()
        calls = []
        hook = net.featEmbedding.spatial_decoder.register_forward_hook(lambda *args: calls.append(1))
        for index in range(6):
            current = {key: value[0, index] for key, value in data.items()}
            result = stream.stream_step('video', 'pair', current, metadata(index * 30, (index + 1) * 30))
            self.assertEqual(len(stream.caches[('video', 'pair')]), min(3, index + 1))
            cached = tuple(torch.cat([item[component] for item in stream.caches[('video', 'pair')]], 1)
                           for component in range(3))
            self.assertEqual(cached[0].shape[1], min(index + 1, 3))
            with torch.no_grad():
                offline = net({key: value[:, max(0, index - 2):index + 1] for key, value in data.items()},
                              torch.tensor([min(3, index + 1)]))[0]
            torch.testing.assert_close(torch.tensor(result['scores']), offline, atol=2e-5, rtol=2e-5)
        self.assertEqual(len(calls), 12)
        hook.remove()
        current = {key: value[0, 0] for key, value in data.items()}
        stream.stream_step('other', 'pair', current, metadata(0, 30))
        stream.stream_step('video', 'second', current, metadata(150, 180))
        self.assertEqual(len(stream.caches), 3)
        stream.reset('video')
        self.assertEqual(list(stream.caches), [('other', 'pair')])
        stream.reset()
        self.assertFalse(stream.caches or stream.last_end or stream.clocks or stream.association.active)

    def test_association_topk_parallel_reopen_gap_flush_format(self):
        assoc = OnlineAssociation(['a', 'b', 'c'], clip_top_n=2)
        first = assoc.step('v', 'p', [.8, .7, .01], metadata(0, 30), 30)
        self.assertEqual([item['event'] for item in first], ['start', 'start'])
        second = assoc.step('v', 'p', [.5, .1, .2], metadata(30, 60), 60)
        self.assertEqual([item['event'] for item in second], ['update', 'end', 'start'])
        self.assertEqual(second[1]['end_fid'], 30)
        self.assertEqual(second[1]['emitted_at_fid'], 60)
        self.assertEqual(first[0]['end_fid'], 30)
        third = assoc.step('v', 'p', [.1, .8, .2], metadata(60, 90), 90)
        self.assertEqual([item['event'] for item in third], ['end', 'start', 'update'])
        gap = assoc.step('v', 'p', [.9, .8, .1], metadata(120, 150), 150)
        self.assertEqual([item['event'] for item in gap], ['start', 'end', 'start', 'end'])
        assoc.flush('v', 160)
        final = assoc.final('v')
        self.assertEqual(len(final), 6)
        for item in final:
            self.assertEqual(len(item['sub_traj']), item['duration'][1] - item['duration'][0])
            self.assertEqual(len(item['obj_traj']), len(item['sub_traj']))
            json.dumps(item)
        merged = next(item for item in final if item['duration'] == [0, 60])
        self.assertAlmostEqual(merged['score'], .8 * .9 * .65)
        scores, _, _ = eval_detection_scores(final, final, .5)
        self.assertEqual(len(scores), len(final))

    def test_stream_gap_and_prefix_consistency(self):
        net = model()
        data = features(6)
        def replay(count):
            engine = StreamPredictor(net)
            outputs = []
            for index in range(count):
                begin = index * 30 + (30 if index >= 3 else 0)
                outputs.append(engine.stream_step('v', 'p', {key: value[0, index] for key, value in data.items()},
                                                  metadata(begin, begin + 30)))
                if index == 3:
                    self.assertEqual(len(engine.caches[('v', 'p')]), 1)
            return outputs
        prefix, full = replay(4), replay(6)
        self.assertEqual(prefix, full[:4])
        for result in full:
            self.assertGreaterEqual(result['emitted_at_fid'], result['end_fid'])
            for event in result['events']:
                self.assertGreaterEqual(event['emitted_at_fid'], event['end_fid'])
        engine = StreamPredictor(net)
        with self.assertRaises(ValueError):
            engine.stream_step('v', 'p', {key: value[0, 0] for key, value in data.items()}, metadata(0, 30), 29)

    def test_training_gradients_and_ignore(self):
        net = model().train()
        data = features(3)
        data = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in data.items()}
        labels = dict(pre_label=torch.tensor([[1., 0, 1, 0], [0., 0, 0, 0]]),
                      sbj_label=torch.tensor([-1, -1]), obj_label=torch.tensor([-1, -1]))
        loss = net(data, torch.tensor([1, 3]), labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (net.featEmbedding.spatial_decoder.transformer, net.featEmbedding.temporal_decoder.transformer,
                       net.featEmbedding.temporal_decoder.intHead):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0)
        self.assertIsNone(net.text_predicate.grad)

    def test_original_loss_reduction_and_object_gradients(self):
        net = model()
        data = features(3)
        data = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in data.items()}
        lengths = torch.tensor([1, 3])
        labels = dict(pre_label=torch.tensor([[1., 0, 1, 0], [0., 0, 0, 0]]),
                      sbj_label=torch.tensor([0, 1]), obj_label=torch.tensor([1, 2]))
        loss = net(data, lengths, labels)
        pre_logits, int_logits, sbj_logits, obj_logits = net.classify_current(net.encode_spatial(data), lengths, 'all')
        expected = sum(focal(pre_logits[index], labels['pre_label'][index]) for index in range(2))
        expected += net.int_loss_weight * sum(focal(int_logits[index], torch.tensor(float(index == 0))) for index in range(2))
        expected += net.obj_loss_weight * (torch.nn.functional.cross_entropy(sbj_logits, labels['sbj_label']) +
                                           torch.nn.functional.cross_entropy(obj_logits, labels['obj_label']))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        gradient = net.featEmbedding.spatial_decoder.objEmb.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        net.src_split = 'base'
        labels['sbj_label'][:] = 1
        labels['obj_label'][:] = 1
        self.assertTrue(torch.isfinite(net(data, lengths, labels)))

    def test_windows_negatives_padding_and_cache_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {key: value[0].numpy() for key, value in features(6, 768).items()}
            with (root / 'video_with_underscores_000012.pkl').open('wb') as handle:
                pickle.dump([arrays, [0, 1, [[0], [], [1], [], [2], []]]], handle)
            dataset = WindowDataset(root, 4, 4)
            self.assertEqual(len(dataset), 6)
            self.assertEqual([len(dataset[index][0]['clip_feat']) for index in range(6)], [1, 2, 3, 4, 4, 4])
            torch.testing.assert_close(dataset[5][0]['clip_feat'], torch.from_numpy(arrays['clip_feat'][2:6]))
            self.assertEqual(dataset[1][1]['pre_label'].sum(), 0)
            _, lengths, targets = padding_collate_fn([dataset[0], dataset[5]])
            self.assertEqual(lengths.tolist(), [1, 4])
            self.assertEqual(targets['pre_label'].shape, (2, 4))
            meta = metadata(5, 170)
            meta['duration'] = [5, 170]
            with (root / 'video_with_underscores_000012.pkl').open('wb') as handle:
                pickle.dump([arrays, meta], handle)
            video, events = next(cached_events(root))
            self.assertEqual(video, 'video_with_underscores')
            self.assertEqual(events[-1][0], 170)
            self.assertEqual(len(events[-1][3]['sbj_traj']), 15)

    def test_checkpoint_reports_and_rejects_stage2(self):
        net = model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            torch.save({'state_dict': net.state_dict(), 'stage': 'stage1'}, path)
            load_checkpoint(net, path)
            state = net.state_dict()
            state['pre_classifier.ctx'] = torch.randn(2)
            torch.save({'state_dict': state}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(net, path)
            state = net.state_dict()
            state.pop('featEmbedding.temporal_decoder.relEmb.weight')
            torch.save({'state_dict': state}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(net, path)

    def test_bad_metadata_does_not_mutate_cache(self):
        net = model()
        engine = StreamPredictor(net)
        data = {key: value[0, 0] for key, value in features(1).items()}
        meta = metadata(0, 30)
        meta['sbj_traj'] = []
        with self.assertRaises(ValueError):
            engine.stream_step('v', 'p', data, meta)
        self.assertFalse(engine.caches)

    def test_split_excludes_predicates_from_topk(self):
        net = model()
        net.tgt_split = 'base'
        engine = StreamPredictor(net)
        data = {key: value[0, 0] for key, value in features(1).items()}
        result = engine.stream_step('v', 'p', data, metadata(0, 30))
        self.assertEqual([event['predicate_id'] for event in result['events']], [0, 2])

    def test_topk_selects_low_scores_and_default_twenty(self):
        assoc = OnlineAssociation([str(index) for index in range(25)])
        events = assoc.step('v', 'p', [index * .0001 for index in range(25)], metadata(0, 30), 30)
        self.assertEqual({event['predicate_id'] for event in events}, set(range(5, 25)))
        assoc = OnlineAssociation(['a', 'b'], clip_top_n=1)
        assoc.step('v', 'p', [.0002, .0001], metadata(0, 30), 30)
        events = assoc.step('v', 'p', [.9, .95], metadata(30, 60), 60)
        self.assertEqual([event['event'] for event in events], ['end', 'start'])
        self.assertEqual(events[0]['end_fid'], 30)
        with self.assertRaises(ValueError):
            OnlineAssociation(['a'], clip_top_n=0)

    def test_matches_original_offline_topk_association(self):
        source = Path(__file__).resolve().parents[2] / 'VidVRD_predication/utils/post_process.py'
        if not source.is_file():
            self.skipTest('optional original project is not present')
        spec = importlib.util.spec_from_file_location('original_post_process', source)
        original = importlib.util.module_from_spec(spec)
        import sys
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(original)
        finally:
            sys.dont_write_bytecode = previous
        from types import SimpleNamespace
        args = SimpleNamespace(clip_len=30, clip_top_n=2, use_prior=False, max_per_video=200)
        scores = torch.tensor([[.03, .02, .01], [.02, .01, .03], [.01, .03, .02], [.03, .02, .01]])
        meta = metadata(0, 120)
        meta['duration'] = [0, 120]
        names = ['a', 'b', 'c']
        expected = original.format_(args, original.association(original.process_pred(args, names, {}, None, scores, meta)))
        online = OnlineAssociation(names, clip_top_n=2)
        for index, current in enumerate(scores):
            online.step('v', 'p', current.tolist(), metadata(index * 30, (index + 1) * 30), (index + 1) * 30)
        online.flush('v', 120)
        actual = online.final('v')
        order = lambda item: (item['triplet'], item['duration'])
        expected, actual = sorted(expected, key=order), sorted(actual, key=order)
        self.assertEqual(len(expected), len(actual))
        for reference, result in zip(expected, actual):
            for key in ('triplet', 'duration', 'sub_traj', 'obj_traj'):
                self.assertEqual(reference[key], result[key])
            self.assertAlmostEqual(reference['score'], result['score'])


if __name__ == '__main__':
    unittest.main()
