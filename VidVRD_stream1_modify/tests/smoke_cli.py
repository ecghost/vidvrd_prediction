import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def main():
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='smoke_', dir=project) as directory:
        root = Path(directory)
        features = root / 'feature'
        for mode in ('train', 'test'):
            (features / f'{mode}_gt_30').mkdir(parents=True)
        info = dict(id2cls={'0': 'a', '1': 'b'}, cls2id={'a': 0, 'b': 1}, cls2split={'a': 'base', 'b': 'novel'})
        (root / 'split.json').write_text(json.dumps(info))
        torch.manual_seed(1)
        text = {key: torch.randn(2, 768) for key in ('subject', 'object', 'predicate', 'objects')}
        torch.save(text, root / 'text.pt')
        arrays = {key: np.zeros(shape, dtype=np.float32) for key, shape in
                  [('clip_feat', (2, 4, 768)), ('bbox_feat', (2, 4, 24)), ('rel_feat', (2, 42)), ('mot_feat', (2, 42))]}
        with (features / 'train_gt_30/synthetic_video_000000.pkl').open('wb') as handle:
            pickle.dump([arrays, [0, 1, [[0, 1], []]]], handle)
        metadata = dict(duration=[0, 45], sbj_cls='a', obj_cls='b', sbj_scr=1., obj_scr=1.,
                        sbj_traj=[[0, 0, 2, 2]] * 45, obj_traj=[[0, 0, 2, 2]] * 45)
        with (features / 'test_gt_30/synthetic_video_000000.pkl').open('wb') as handle:
            pickle.dump([arrays, metadata], handle)
        groundtruth = {'synthetic_video': [dict(triplet=['a', 'a', 'b'], duration=[0, 30],
                                                sub_traj=metadata['sbj_traj'][:30], obj_traj=metadata['obj_traj'][:30]),
                                           dict(triplet=['a', 'b', 'b'], duration=[0, 30],
                                                sub_traj=metadata['sbj_traj'][:30], obj_traj=metadata['obj_traj'][:30])]}
        (root / 'gt.json').write_text(json.dumps(groundtruth))
        common = ['--feature_root', str(features), '--pred_class_split', str(root / 'split.json'),
                  '--obj_class_split', str(root / 'split.json'), '--output_dir', str(root / 'output'),
                  '--annotation', str(root / 'gt.json'),
                  '--device', 'cpu', '--src_split', 'all', '--window_size', '2',
                  '--clip_top_n', '20']
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, 'train_stage1.py', *common, '--text_vectors', str(root / 'text.pt'),
                        '--batch_size', '2', '--max_steps', '1', '--num_epochs', '1'], cwd=project, env=env, check=True)
        checkpoints = list((root / 'output/checkpoint').glob('*.pth'))
        assert len(checkpoints) == 1
        checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
        assert checkpoint['metrics']['novel'][1] == {50: 1., 100: 1.}
        assert checkpoint['metrics']['all'][1] == {50: 1., 100: 1.}
        assert checkpoint['mean_map'] == (checkpoint['metrics']['all'][0] + checkpoint['metrics']['novel'][0]) / 2
        subprocess.run([sys.executable, 'infer_stream.py', *common, '--init_checkpoint', str(checkpoints[0])],
                       cwd=project, env=env, check=True)
        files = list((root / 'output').rglob('*'))
        assert {path.name for path in files if path.is_dir()} == {'log', 'checkpoint'}
        assert all(path.suffix in ('.log', '.pth') for path in files if path.is_file())
        logs = list((root / 'output/log').glob('*.log'))
        assert len(logs) == 2
        for path in logs:
            text = path.read_text()
            assert 'Novel split' in text and 'All split' in text and 'Mean mAP' in text
        assert any('Training loss' in path.read_text() for path in logs)
        print('PASS: 768-d training/inference, all/novel metrics, best checkpoint and log-only outputs')


if __name__ == '__main__':
    main()
