from collections import OrderedDict

from dataset import load_video_events


def scheduled_batches(video_index, clip_len=30, inference_batch_size=32, max_active_videos=1):
    if inference_batch_size < 1 or max_active_videos < 1:
        raise ValueError('inference_batch_size and max_active_videos must be positive')
    remaining = iter(video_index)
    active = OrderedDict()
    exhausted = False
    while active or not exhausted:
        while len(active) < max_active_videos and not exhausted:
            try:
                video, pairs = next(remaining)
            except StopIteration:
                exhausted = True
                break
            active[video] = [load_video_events(pairs, clip_len), 0]
        batch, finished = [], []
        for video, state in active.items():
            events, cursor = state
            if cursor == len(events):
                finished.append((video, events[-1][0] if events else None))
                continue
            clock = events[cursor][0]
            seen = set()
            while cursor < len(events) and len(batch) < inference_batch_size:
                end, pair, features, metadata = events[cursor]
                if end != clock or pair in seen:
                    break
                seen.add(pair)
                batch.append(dict(video_id=video, pair_id=pair, features=features,
                                  metadata=metadata, emitted_at_fid=end))
                cursor += 1
            state[1] = cursor
            if cursor == len(events):
                finished.append((video, events[-1][0]))
            if len(batch) == inference_batch_size:
                break
        if not active:
            break
        yield batch, finished
        batch = None
        for video, _ in finished:
            del active[video]
        events = state = features = metadata = None
        if active:
            active.move_to_end(next(iter(active)))
