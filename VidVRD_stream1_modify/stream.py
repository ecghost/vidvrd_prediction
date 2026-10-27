from collections import deque

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.post_process import OnlineAssociation


class StreamPredictor:
    def __init__(self, model, clip_top_n=30):
        self.model = model.eval()
        names = [model.pred_split['id2cls'][str(index)] for index in range(len(model.pred_split['id2cls']))]
        self.association = OnlineAssociation(names, clip_top_n,
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

    def stream_step(self, video_id, pair_id, features, metadata, emitted_at_fid=None):
        return self.stream_step_batch([dict(video_id=video_id, pair_id=pair_id, features=features,
                                            metadata=metadata, emitted_at_fid=emitted_at_fid)])[0]

    @torch.inference_mode()
    def stream_step_batch(self, events):
        if not events:
            return []
        keys, clocks, video_clocks = set(), [], {}
        for event in events:
            video, pair, meta = event['video_id'], event['pair_id'], event['metadata']
            key = (video, pair)
            clock = event.get('emitted_at_fid')
            clock = meta['end_fid'] if clock is None else clock
            if key in keys:
                raise ValueError('a batch may contain at most one segment per video/pair')
            if video in video_clocks and video_clocks[video] != clock:
                raise ValueError('a batch may contain only one release-time group per video')
            if clock < meta['end_fid'] or clock < self.clocks.get(video, -1):
                raise ValueError('segments must be released in completion-clock order')
            begin, end = meta['begin_fid'], meta['end_fid']
            if end <= begin or len(meta['sbj_traj']) != end - begin or len(meta['obj_traj']) != end - begin:
                raise ValueError('invalid half-open segment duration/trajectory lengths')
            if key in self.last_end and begin < self.last_end[key]:
                raise ValueError('overlapping or out-of-order pair segments')
            keys.add(key)
            clocks.append(clock)
            video_clocks[video] = clock
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = {name: torch.stack([event['features'][name] for event in events]).to(device).unsqueeze(1)
                  for name in events[0]['features']}
        spatial = self.model.encode_spatial(inputs)
        histories, gaps = [], []
        for index, (event, clock) in enumerate(zip(events, clocks)):
            video, pair, meta = event['video_id'], event['pair_id'], event['metadata']
            key = (video, pair)
            gap = key in self.last_end and meta['begin_fid'] != self.last_end[key]
            if gap:
                self.caches.pop(key, None)
            gaps.append(gap)
            cache = self.caches.setdefault(key, deque(maxlen=self.model.window_size))
            cache.append(tuple(value[index:index + 1].clone() for value in spatial))
            histories.append(tuple(torch.cat([item[component] for item in cache], dim=1).squeeze(0)
                                   for component in range(3)))
        lengths = torch.tensor([len(history[0]) for history in histories], device=device)
        window = tuple(pad_sequence([history[component] for history in histories], batch_first=True)
                       for component in range(3))
        scores = self.model.scores(self.model.classify_current(window, lengths, self.model.tgt_split)).cpu().tolist()
        results = []
        for event, clock, row, gap in zip(events, clocks, scores, gaps):
            video, pair, meta = event['video_id'], event['pair_id'], event['metadata']
            closed = self.association.flush(video, clock, pair) if gap else []
            closed += self.association.step(video, pair, row, meta, clock)
            self.last_end[(video, pair)], self.clocks[video] = meta['end_fid'], clock
            results.append(dict(video_id=video, pair_id=pair, begin_fid=meta['begin_fid'], end_fid=meta['end_fid'],
                                emitted_at_fid=clock, scores=row, events=closed))
        return results

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
