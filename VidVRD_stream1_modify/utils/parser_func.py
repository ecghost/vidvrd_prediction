import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Stage 1 cached-feature streaming GT PredCls')
    for name in ('dataset_root', 'feature_root', 'annotation', 'pred_class_split', 'obj_class_split',
                 'prior', 'text_vectors', 'text_model_weights', 'init_checkpoint'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output_dir', type=Path, default=Path('output'))
    parser.add_argument('--clip_len', type=int, default=30)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--inference_batch_size', type=int, default=32)
    parser.add_argument('--max_active_videos', type=int, default=1)
    parser.add_argument('--max_videos', type=int, default=0, help='0 evaluates all indexed videos')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--src_split', choices=['base', 'novel', 'all'], default='base')
    parser.add_argument('--tgt_split', choices=['base', 'novel', 'all'], default='all')
    parser.add_argument('--clip_top_n', type=int, default=20,
                        help='top predicates per completed pair segment; no score threshold')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--obj_loss_weight', type=float, default=0.2)
    parser.add_argument('--int_loss_weight', type=float, default=0.1)
    parser.add_argument('--limit_pairs', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--max_per_video', type=int, default=200)
    parser.add_argument('--progress_width', type=int, default=100, help='tqdm total width in columns')
    parser.add_argument('--allow_partial_init', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.window_size <= 40 or args.clip_len <= 0 or args.batch_size <= 0:
        parser.error('window_size must be 1..40, clip_len and batch_size positive')
    if args.clip_top_n < 1:
        parser.error('clip_top_n must be positive')
    if min(args.inference_batch_size, args.max_active_videos) < 1 or args.max_videos < 0:
        parser.error('inference batching parameters must be positive; max_videos must be nonnegative')
    if args.progress_width < 40:
        parser.error('progress_width must be at least 40')
    if args.num_epochs < 1 or args.learning_rate <= 0 or min(args.limit_pairs, args.max_steps, args.max_per_video) < 0:
        parser.error('epochs and learning_rate must be positive; limits must be nonnegative')
    if args.prior:
        parser.error('this baseline preserves raw PredCls scores; prior fusion is intentionally disabled')
    if args.dataset_root:
        args.feature_root = args.feature_root or args.dataset_root / 'feature'
        args.annotation = args.annotation or args.dataset_root / 'data/test_relation_gt.json'
        args.pred_class_split = args.pred_class_split or args.dataset_root / 'data/openvoc_pred_class_spilt_info.json'
        args.obj_class_split = args.obj_class_split or args.dataset_root / 'data/openvoc_obj_class_spilt_info.json'
    if not args.feature_root or not args.pred_class_split or not args.obj_class_split:
        parser.error('provide dataset_root or feature_root and both class-split paths')
    if not args.init_checkpoint and not args.text_vectors and not args.text_model_weights:
        parser.error('provide offline text_vectors, local text_model_weights, or a native streaming checkpoint')
    args.output_dir = args.output_dir.resolve()
    project = Path(__file__).resolve().parents[1]
    if not args.output_dir.is_relative_to(project):
        parser.error('output_dir must be inside VidVRD_stream1')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args
