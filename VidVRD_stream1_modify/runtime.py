import json
import random

import numpy as np
import torch
from tqdm import tqdm

from dataset import index_videos
from model_zoo.stage1 import Model, load_checkpoint
from stream import StreamPredictor
from replay import scheduled_batches
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


@torch.inference_mode()
def replay(model, args):
    predictor = StreamPredictor(model, args.clip_top_n)
    predictions = {}
    videos = index_videos(args.feature_root / f'test_gt_{args.clip_len}', args.limit_pairs, args.max_videos)
    batches = scheduled_batches(videos, args.clip_len, args.inference_batch_size, args.max_active_videos)
    with tqdm(total=len(videos), desc=f'Infer {model.tgt_split}', unit='video', ncols=args.progress_width) as progress:
        for events, finished in batches:
            predictor.stream_step_batch(events)
            for video, clock in finished:
                if clock is not None:
                    predictor.finish(video, clock)
                predictions[video] = predictor.association.final(video, args.max_per_video)
                predictor.reset(video)
                progress.update(1)
            del events, finished
    if not predictions:
        raise ValueError('no test cache files found')
    metrics = None
    if args.annotation and args.annotation.is_file():
        from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc
        metrics = eval_relation_detection_openvoc(
            target_split_pred=model.tgt_split, prediction_results=predictions,
            enti_cls_split_info_path=str(args.obj_class_split), pred_cls_split_info_path=str(args.pred_class_split),
            gt_json=str(args.annotation), video_ids=[video for video, _ in videos] if args.max_videos else None)
        metrics = (float(metrics[0]), {key: float(value) for key, value in metrics[1].items()})
    return predictions, metrics


def evaluate_splits(model, args, logger):
    if not args.annotation or not args.annotation.is_file():
        raise ValueError('all/novel evaluation requires an existing --annotation GT JSON')
    original_split = model.tgt_split
    metrics = {}
    try:
        for split in ('novel', 'all'):
            model.tgt_split = split
            _, result = replay(model, args)
            metrics[split] = result
            logger.info('PredCls and %s split | mAP:%.4f, Recall@50:%.4f, Recall@100:%.4f',
                        split.title(), result[0] * 100, result[1][50] * 100, result[1][100] * 100)
    finally:
        model.tgt_split = original_split
    mean_map = (metrics['novel'][0] + metrics['all'][0]) / 2
    logger.info('Mean mAP: %.4f', mean_map * 100)
    return metrics, mean_map
