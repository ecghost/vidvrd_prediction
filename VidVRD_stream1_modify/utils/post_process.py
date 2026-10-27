from copy import deepcopy

import numpy as np


class OnlineAssociation:
    def __init__(self, predicates, clip_top_n=20, predicate_ids=None):
        if not isinstance(clip_top_n, int) or clip_top_n < 1:
            raise ValueError('clip_top_n must be a positive integer')
        self.predicates = predicates
        self.predicate_ids = set(range(len(predicates)) if predicate_ids is None else predicate_ids)
        self.clip_top_n = clip_top_n
        self.active, self.finished = {}, {}

    def reset(self, video_id=None):
        if video_id is None:
            self.active.clear()
            self.finished.clear()
        else:
            self.active = {key: value for key, value in self.active.items() if key[0] != video_id}
            self.finished.pop(video_id, None)

    def snapshot(self, key, state, kind, clock):
        if clock < state['duration'][1]:
            raise ValueError('emission before segment completion')
        return deepcopy(dict(event=kind, video_id=key[0], pair_id=key[1], predicate_id=key[2],
                             start_fid=state['duration'][0], end_fid=state['duration'][1],
                             score_sum=state['score_sum'], num_segments=state['num_segments'],
                             predicate_score=state['score_sum'] / state['num_segments'],
                             score=state['score'], emitted_at_fid=clock, active=kind != 'end',
                             triplet=state['triplet']))

    def close(self, key, clock):
        state = self.active.pop(key)
        event = self.snapshot(key, state, 'end', clock)
        result = {field: deepcopy(state[field]) for field in ('triplet', 'score', 'duration', 'sub_traj', 'obj_traj')}
        self.finished.setdefault(key[0], []).append(result)
        return event

    def flush(self, video_id, clock, pair_id=None):
        return [self.close(key, clock) for key in list(self.active)
                if key[0] == video_id and (pair_id is None or key[1] == pair_id)]

    def step(self, video_id, pair_id, scores, meta, clock):
        begin, end = meta['begin_fid'], meta['end_fid']
        if clock < end or end <= begin:
            raise ValueError('invalid segment release time')
        if len(meta['sbj_traj']) != end - begin or len(meta['obj_traj']) != end - begin:
            raise ValueError('trajectory length must match half-open duration')
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(self.predicates),) or not np.isfinite(scores).all():
            raise ValueError('scores must be a finite vector matching predicate count')
        eligible = np.array(sorted(self.predicate_ids), dtype=np.int64)
        selected = set(eligible[np.argsort(-scores[eligible])[:self.clip_top_n]].tolist())
        events = []
        for predicate, score in enumerate(scores):
            if predicate not in self.predicate_ids:
                continue
            score = float(score)
            key = (video_id, pair_id, predicate)
            state = self.active.get(key)
            if state and state['duration'][1] != begin:
                events.append(self.close(key, clock))
                state = None
            if state and predicate not in selected:
                events.append(self.close(key, clock))
            elif predicate in selected:
                kind = 'update' if state else 'start'
                if state is None:
                    state = dict(triplet=[meta['sbj_cls'], self.predicates[predicate], meta['obj_cls']],
                                 duration=[begin, begin], sub_traj=[], obj_traj=[], score_sum=0.,
                                 num_segments=0, object_score=float(meta['sbj_scr']) * float(meta['obj_scr']))
                    self.active[key] = state
                state['duration'][1] = end
                state['sub_traj'].extend(deepcopy(meta['sbj_traj']))
                state['obj_traj'].extend(deepcopy(meta['obj_traj']))
                state['score_sum'] += score
                state['num_segments'] += 1
                state['score'] = state['object_score'] * state['score_sum'] / state['num_segments']
                events.append(self.snapshot(key, state, kind, clock))
        return events

    def final(self, video_id, top_k=200):
        return sorted(deepcopy(self.finished.get(video_id, [])), key=lambda item: item['score'], reverse=True)[:top_k]
