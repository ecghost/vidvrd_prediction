import json
import random

import numpy as np
import torch

from dataset import cached_events
from model_zoo.stage1 import Model, load_checkpoint
from stream import StreamPredictor
from text_encoder import build_text_vectors


def make_model(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pred_info = json.loads(args.pred_class_split.read_text())
    obj_info = json.loads(args.obj_class_split.read_text())
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False) if args.init_checkpoint else None
    if checkpoint and checkpoint.get('format') == 'stream1':
        if checkpoint['window_size'] != args.window_size or checkpoint['clip_len'] != args.clip_len:
            raise ValueError('checkpoint window_size/clip_len mismatch')
        if checkpoint['pred_split'] != pred_info or checkpoint['obj_split'] != obj_info:
            raise ValueError('checkpoint class definitions differ from supplied splits')
        text = checkpoint['text']
    elif args.text_vectors:
        text = torch.load(args.text_vectors, map_location='cpu', weights_only=True)
    elif args.text_model_weights:
        text = build_text_vectors(pred_info, obj_info, args.text_model_weights, args.device)
    else:
        raise ValueError('legacy Stage 1 checkpoint excludes fixed text buffers: supply text_vectors or text_model_weights')
    model = Model(text, pred_info, obj_info, args.window_size, args.src_split, args.tgt_split,
                  args.obj_loss_weight, args.int_loss_weight)
    if args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint, args.allow_partial_init)
    return model.to(args.device)


def replay(model, args, output_name='predictions'):
    predictor = StreamPredictor(model, args.clip_top_n)
    predictions = {}
    with open(args.output_dir / f'{output_name}.events.jsonl', 'w') as events_file, \
            open(args.output_dir / f'{output_name}.segments.jsonl', 'w') as segments_file:
        for video, events in cached_events(args.feature_root / f'test_gt_{args.clip_len}', args.clip_len, args.limit_pairs):
            predictor.reset(video)
            for clock, pair, features, metadata in events:
                result = predictor.stream_step(video, pair, features, metadata, clock)
                segments_file.write(json.dumps({key: value for key, value in result.items() if key != 'events'}) + '\n')
                for event in result['events']:
                    events_file.write(json.dumps(event) + '\n')
            if events:
                for event in predictor.finish(video, events[-1][0]):
                    events_file.write(json.dumps(event) + '\n')
            predictions[video] = predictor.association.final(video, args.max_per_video)
            predictor.reset(video)
    if not predictions:
        raise ValueError('no test cache files found')
    (args.output_dir / f'{output_name}.json').write_text(json.dumps(predictions))
    metrics = None
    if args.annotation and args.annotation.is_file():
        from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc
        metrics = eval_relation_detection_openvoc(
            target_split_pred=args.tgt_split, prediction_results=predictions,
            enti_cls_split_info_path=str(args.obj_class_split), pred_cls_split_info_path=str(args.pred_class_split),
            gt_json=str(args.annotation))
        metrics = (float(metrics[0]), {key: float(value) for key, value in metrics[1].items()})
        print('mAP, recall:', metrics)
        (args.output_dir / f'{output_name}.metrics.json').write_text(json.dumps(metrics))
    return predictions, metrics
