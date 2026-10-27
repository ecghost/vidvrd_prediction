import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

FEATURES = ('clip_feat', 'bbox_feat', 'rel_feat', 'mot_feat')


def read_pair(path):
    with open(path, 'rb') as handle:
        features, metadata = pickle.load(handle)
    features = {key: np.asarray(features[key], dtype=np.float32) for key in FEATURES}
    length = len(features['clip_feat'])
    if any(len(value) != length for value in features.values()):
        raise ValueError(f'inconsistent feature lengths: {path}')
    for key, shape in [('clip_feat', (4, 768)), ('bbox_feat', (4, 24)),
                       ('rel_feat', (42,)), ('mot_feat', (42,))]:
        if length and features[key].shape[1:] != shape:
            raise ValueError(f'{path}: invalid {key} shape {features[key].shape}')
    return features, metadata


class WindowDataset(Dataset):
    def __init__(self, root, num_predicates, window_size=4, limit_pairs=0):
        if window_size < 1:
            raise ValueError('window_size must be positive')
        self.paths = sorted(Path(root).glob('*.pkl'))
        if limit_pairs:
            self.paths = self.paths[:limit_pairs]
        self.window_size, self.num_predicates = window_size, num_predicates
        self.index = []
        for pair_index, path in enumerate(self.paths):
            features, metadata = read_pair(path)
            if isinstance(metadata, dict) or len(metadata[2]) != len(features['clip_feat']):
                raise ValueError(f'expected training labels in {path}')
            self.index.extend((pair_index, current) for current in range(len(features['clip_feat'])))
        self._read = lru_cache(maxsize=8)(read_pair)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        pair_index, current = self.index[index]
        features, labels = self._read(self.paths[pair_index])
        start = max(0, current + 1 - self.window_size)
        inputs = {key: torch.from_numpy(value[start:current + 1]) for key, value in features.items()}
        predicate = torch.zeros(self.num_predicates)
        predicate[labels[2][current]] = 1
        return inputs, {'pre_label': predicate, 'sbj_label': int(labels[0]), 'obj_label': int(labels[1])}


def padding_collate_fn(batch):
    inputs, labels = zip(*batch)
    lengths = torch.tensor([len(item['clip_feat']) for item in inputs])
    features = {key: pad_sequence([item[key] for item in inputs], batch_first=True) for key in FEATURES}
    targets = {key: torch.stack([torch.as_tensor(item[key]) for item in labels]) for key in labels[0]}
    return features, lengths, targets


def index_videos(root, limit_pairs=0, max_videos=0):
    paths = sorted(Path(root).glob('*.pkl'))
    if limit_pairs:
        paths = paths[:limit_pairs]
    videos = {}
    for path in paths:
        match = re.fullmatch(r'(.+)_([0-9]{6})', path.stem)
        if not match:
            raise ValueError(f'unrecognized cache filename: {path.name}')
        video, pair = match.groups()
        videos.setdefault(video, []).append((pair, path))
    items = sorted(videos.items())
    return items[:max_videos] if max_videos else items


def load_video_events(pairs, clip_len):
    if clip_len < 1:
        raise ValueError('clip_len must be positive')
    events = []
    for pair, path in pairs:
        features, meta = read_pair(path)
        if not isinstance(meta, dict):
            raise ValueError('replay requires test cache with real duration/trajectories')
        begin, end = meta['duration']
        count = len(features['clip_feat'])
        expected = (end - begin + clip_len - 1) // clip_len
        if count != expected or len(meta['sbj_traj']) != end - begin or len(meta['obj_traj']) != end - begin:
            raise ValueError(f'duration/feature/trajectory mismatch: {path}')
        for index in range(count):
            first, last = begin + index * clip_len, min(end, begin + (index + 1) * clip_len)
            current = {key: torch.from_numpy(value[index]) for key, value in features.items()}
            metadata = {key: meta[key] for key in ('sbj_cls', 'obj_cls', 'sbj_scr', 'obj_scr')}
            metadata.update(sbj_traj=meta['sbj_traj'][first - begin:last - begin],
                            obj_traj=meta['obj_traj'][first - begin:last - begin],
                            begin_fid=first, end_fid=last)
            events.append((last, pair, current, metadata))
    return sorted(events, key=lambda event: (event[0], event[1]))


def cached_events(root, clip_len=30, limit_pairs=0, max_videos=0):
    for video, pairs in index_videos(root, limit_pairs, max_videos):
        yield video, load_video_events(pairs, clip_len)
