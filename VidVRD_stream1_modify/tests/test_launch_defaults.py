import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class LaunchDefaultsTests(unittest.TestCase):
    def launch(self, script, overrides=None, arguments=()):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project) as directory:
            executable = Path(directory) / 'python'
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$@"\n')
            executable.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if key not in
                   ('DATASET_ROOT', 'TEXT_VECTORS', 'TEXT_MODEL_WEIGHTS', 'CHECKPOINT', 'DEVICE', 'CUDA_VISIBLE_DEVICES')}
            env['PATH'] = directory + os.pathsep + env['PATH']
            env.update(overrides or {})
            return subprocess.check_output(['bash', str(project / script), *arguments], env=env, text=True).splitlines()

    def test_train_defaults_match_original_stage1(self):
        args = self.launch('train.sh')
        for name, expected in [('--batch_size', '32'), ('--num_epochs', '40'), ('--int_loss_weight', '0.1'),
                               ('--device', 'cuda:0'), ('--learning_rate', '1e-4')]:
            self.assertEqual(args[args.index(name) + 1], expected)
        self.assertIn('CUDA_VISIBLE_DEVICES=2', args)
        self.assertTrue(Path(args[args.index('--dataset_root') + 1]).is_dir())
        self.assertTrue(Path(args[args.index('--text_model_weights') + 1]).is_file())

    def test_infer_default_checkpoint(self):
        args = self.launch('infer.sh')
        self.assertTrue(args[args.index('--init_checkpoint') + 1].endswith('/output/checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth'))

    def test_training_output_isolated_and_overridable(self):
        first = self.launch('train.sh')
        second = self.launch('train.sh')
        first_path = first[first.index('--output_dir') + 1]
        second_path = second[second.index('--output_dir') + 1]
        self.assertTrue(first_path.startswith('output/run_'))
        self.assertNotEqual(first_path, second_path)
        custom = self.launch('train.sh', {'OUTPUT_DIR': 'output/custom'})
        self.assertEqual(custom[custom.index('--output_dir') + 1], 'output/custom')

    def test_overrides(self):
        args = self.launch('train.sh', {'TEXT_VECTORS': '/custom/text.pt', 'DEVICE': 'cpu',
                                      'DATASET_ROOT': '/custom/data', 'CUDA_VISIBLE_DEVICES': '0'},
                           ['--batch_size', '2'])
        self.assertIn('/custom/text.pt', args)
        self.assertNotIn('--text_model_weights', args)
        self.assertEqual(args[-2:], ['--batch_size', '2'])
        self.assertIn('CUDA_VISIBLE_DEVICES=0', args)


if __name__ == '__main__':
    unittest.main()
