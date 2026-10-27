import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import cached_events, index_videos
from reference_serial import StreamPredictor as SerialPredictor
from replay import scheduled_batches
from runtime import make_model
from stream import StreamPredictor
from utils.parser_func import parse_args
from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc


def main():
    args = parse_args()
    if not args.max_videos:
        raise ValueError('set --max_videos to bound this correctness comparison')
    net = make_model(args).eval()
    root = args.feature_root / f'test_gt_{args.clip_len}'
    videos = index_videos(root, args.limit_pairs, args.max_videos)
    for split in ('novel', 'all'):
        net.tgt_split = split
        serial = SerialPredictor(net, args.clip_top_n)
        reference, reference_segments = {}, {}
        for video, events in cached_events(root, args.clip_len, args.limit_pairs, args.max_videos):
            for clock, pair, feature, meta in events:
                reference_segments[(video, pair, clock)] = serial.stream_step(video, pair, feature, meta, clock)
            if events:
                serial.finish(video, events[-1][0])
            reference[video] = serial.association.final(video, args.max_per_video)
            serial.reset(video)
        def metrics(predictions):
            return eval_relation_detection_openvoc(target_split_pred=split, prediction_results=predictions,
                enti_cls_split_info_path=str(args.obj_class_split), pred_cls_split_info_path=str(args.pred_class_split),
                gt_json=str(args.annotation), video_ids=[video for video, _ in videos])
        baseline = metrics(reference)
        for batch_size, active in [(1, 1), (args.inference_batch_size, 1),
                                    (args.inference_batch_size, args.max_active_videos)]:
            engine = StreamPredictor(net, args.clip_top_n)
            final = {}
            for items, finished in scheduled_batches(videos, args.clip_len, batch_size, active):
                for result in engine.stream_step_batch(items):
                    old = reference_segments[(result['video_id'], result['pair_id'], result['emitted_at_fid'])]
                    torch.testing.assert_close(torch.tensor(old['scores']), torch.tensor(result['scores']), atol=2e-5, rtol=2e-5)
                    assert len(old['events']) == len(result['events'])
                    for left, right in zip(old['events'], result['events']):
                        for key in left:
                            if key in ('score', 'score_sum', 'predicate_score'):
                                assert abs(left[key] - right[key]) <= 2e-5 * max(1, abs(left[key]))
                            else:
                                assert left[key] == right[key], (key, left, right)
                for video, clock in finished:
                    if clock is not None:
                        engine.finish(video, clock)
                    final[video] = engine.association.final(video, args.max_per_video)
                    engine.reset(video)
                del items
            order = lambda item: (item['triplet'], item['duration'])
            for video in reference:
                assert len(reference[video]) == len(final[video])
                for old, new in zip(sorted(reference[video], key=order), sorted(final[video], key=order)):
                    assert {key: value for key, value in old.items() if key != 'score'} == {key: value for key, value in new.items() if key != 'score'}
                    assert abs(old['score'] - new['score']) <= 2e-5
            actual = metrics(final)
            assert abs(float(actual[0]) - float(baseline[0])) <= 1e-7
            assert all(abs(float(actual[1][key]) - float(baseline[1][key])) <= 1e-7 for key in baseline[1])
            print(f'PASS {split} batch={batch_size} active={active}: mAP/Recall={actual}')


if __name__ == '__main__':
    main()
