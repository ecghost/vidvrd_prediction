import logging

import torch
from tqdm import tqdm


def run_name(args):
    return (f'rela_vidvrd_bs{args.batch_size}_lr{args.learning_rate}_dim512_None_'
            'rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1')


class ProgressHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


def setup_logger(args, inference=False):
    folder = args.output_dir / 'log'
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('stream1')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    path = folder / (run_name(args) + ('_infer' if inference else '') + '.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.FileHandler(path, mode='a'), ProgressHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info('Configuration: %s', vars(args))
    return logger


def checkpoint_path(args):
    return args.output_dir / 'checkpoint' / (run_name(args) + '.pth')


def save_best(checkpoint, path, mean_map, best_map):
    if mean_map <= best_map:
        return best_map
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_checkpoint = dict(checkpoint, mean_map=mean_map)
    torch.save(torch_checkpoint, path)
    return mean_map
