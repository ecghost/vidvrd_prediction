from collections import deque

import torch

from utils.post_process import OnlineAssociation


class StreamPredictor:
    def __init__(self, model, on_threshold=0.5, off_threshold=0.3):
        self.model = model.eval()
        names = [model.pred_split['id2cls'][str(index)] for index in range(len(model.pred_split['id2cls']))]
        self.association = OnlineAssociation(names, on_threshold, off_threshold,
                                             model.ids(model.pred_split, model.tgt_split))
        self.caches, self.last_end, self.clocks = {}, {}, {}

    def reset(self, video_id=None):
        if video_id is None:
            self.caches.clear()
            self.last_end.clear()
            self.clocks.clear()
        else:
            self.caches = {key: value for key, value in self.caches.items() if key[0] != video_id}
            self.last_end = {key: value for key, value in self.last_end.items() if key[0] != video_id}
            self.clocks.pop(video_id, None)
        self.association.reset(video_id)

    @torch.no_grad()
    def stream_step(self, video_id, pair_id, features, metadata, emitted_at_fid=None):
        clock = metadata['end_fid'] if emitted_at_fid is None else emitted_at_fid
        if clock < metadata['end_fid'] or clock < self.clocks.get(video_id, -1):
            raise ValueError('segments must be released in completion-clock order')
        key = (video_id, pair_id)
        begin, end = metadata['begin_fid'], metadata['end_fid']
        if end <= begin or len(metadata['sbj_traj']) != end - begin or len(metadata['obj_traj']) != end - begin:
            raise ValueError('invalid half-open segment duration/trajectory lengths')
        if key in self.last_end and begin < self.last_end[key]:
            raise ValueError('overlapping or out-of-order pair segments')
        events = []
        if key in self.last_end and begin != self.last_end[key]:
            events += self.association.flush(video_id, clock, pair_id)
            self.caches.pop(key, None)
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device).unsqueeze(0).unsqueeze(0) for name, value in features.items()}
        spatial = self.model.encode_spatial(inputs)
        cache = self.caches.setdefault(key, deque(maxlen=self.model.window_size))
        cache.append(tuple(value.detach() for value in spatial))
        window = tuple(torch.cat([item[index] for item in cache], dim=1) for index in range(3))
        lengths = torch.tensor([len(cache)], device=device)
        scores = self.model.scores(self.model.classify_current(window, lengths, self.model.tgt_split))[0]
        events += self.association.step(video_id, pair_id, scores.cpu().tolist(), metadata, clock)
        self.last_end[key], self.clocks[video_id] = end, clock
        return {'video_id': video_id, 'pair_id': pair_id, 'begin_fid': begin, 'end_fid': end,
                'emitted_at_fid': clock, 'scores': scores.cpu().tolist(), 'events': events}

    def finish(self, video_id, clock, pair_id=None):
        if clock < self.clocks.get(video_id, -1):
            raise ValueError('finish clock cannot precede observed segments')
        events = self.association.flush(video_id, clock, pair_id)
        keys = [key for key in self.caches if key[0] == video_id and (pair_id is None or key[1] == pair_id)]
        for key in keys:
            del self.caches[key]
            self.last_end.pop(key, None)
        self.clocks[video_id] = clock
        return events
