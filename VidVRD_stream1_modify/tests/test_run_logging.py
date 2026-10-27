import tempfile
import unittest
from pathlib import Path

import torch

from utils.run_logging import save_best


class BestCheckpointTests(unittest.TestCase):
    def test_only_strict_improvements_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint/best.pth'
            best = -float('inf')
            for epoch, score in enumerate([0., .3, .2, .3, .4], 1):
                best = save_best({'epoch': epoch}, path, score, best)
                expected_epoch = [1, 2, 2, 2, 5][epoch - 1]
                saved = torch.load(path, weights_only=True)
                self.assertEqual(saved['epoch'], expected_epoch)
                self.assertEqual(saved['mean_map'], best)
            self.assertEqual(len(list(path.parent.iterdir())), 1)
