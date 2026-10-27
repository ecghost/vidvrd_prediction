import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runtime import evaluate_splits, replay


class EvaluationSplitTests(unittest.TestCase):
    def test_replay_evaluates_model_split_not_cli_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = dict(id2cls={'0': 'base_rel', '1': 'novel_rel'},
                        cls2split={'base_rel': 'base', 'novel_rel': 'novel'})
            objects = dict(id2cls={'0': 'subject', '1': 'object'},
                           cls2split={'subject': 'base', 'object': 'base'})
            (root / 'pred.json').write_text(json.dumps(info))
            (root / 'obj.json').write_text(json.dumps(objects))
            def relation(predicate):
                return dict(triplet=['subject', predicate, 'object'], score=1., duration=[0, 30],
                            sub_traj=[[0, 0, 2, 2]] * 30, obj_traj=[[0, 0, 2, 2]] * 30)
            (root / 'gt.json').write_text(json.dumps({'video': [relation('base_rel'), relation('novel_rel')]}))
            args = SimpleNamespace(tgt_split='all', clip_top_n=20, feature_root=root, clip_len=30,
                                   limit_pairs=0, max_videos=0, inference_batch_size=2, max_active_videos=1,
                                   progress_width=80, max_per_video=200, annotation=root / 'gt.json',
                                   pred_class_split=root / 'pred.json', obj_class_split=root / 'obj.json')
            model = SimpleNamespace(tgt_split='novel')
            def predictor(current_model, clip_top_n):
                instance = Mock()
                selected = ['base_rel', 'novel_rel'] if current_model.tgt_split == 'all' else [current_model.tgt_split + '_rel']
                instance.association.final.return_value = [relation(name) for name in selected]
                return instance
            with patch('runtime.StreamPredictor', side_effect=predictor), \
                    patch('runtime.index_videos', return_value=[('video', [])]), \
                    patch('runtime.scheduled_batches', side_effect=lambda *args: iter([([], [('video', None)])])):
                for split in ('novel', 'base', 'all'):
                    model.tgt_split = split
                    _, metrics = replay(model, args)
                    self.assertAlmostEqual(metrics[0], 1., msg=split)
                    self.assertEqual(metrics[1], {50: 1., 100: 1.}, split)
                model.tgt_split = 'base'
                metrics, mean_map = evaluate_splits(model, args, logging.getLogger('test_eval_splits'))
                self.assertEqual(set(metrics), {'novel', 'all'})
                self.assertEqual(mean_map, 1.)
                self.assertEqual(model.tgt_split, 'base')
                self.assertEqual(args.tgt_split, 'all')

    def test_evaluation_failure_restores_original_model_split(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / 'gt.json'
            annotation.write_text('{}')
            model = SimpleNamespace(tgt_split='base')
            with patch('runtime.replay', side_effect=RuntimeError('evaluation failed')):
                with self.assertRaises(RuntimeError):
                    evaluate_splits(model, SimpleNamespace(annotation=annotation), logging.getLogger('test'))
            self.assertEqual(model.tgt_split, 'base')
