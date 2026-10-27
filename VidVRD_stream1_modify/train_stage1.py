import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WindowDataset, padding_collate_fn
from runtime import make_model, evaluate_splits
from utils.parser_func import parse_args
from utils.run_logging import checkpoint_path, save_best, setup_logger


def main():
    args = parse_args()
    logger = setup_logger(args)
    if not args.annotation or not args.annotation.is_file():
        raise ValueError('training model selection requires an existing --annotation GT JSON')
    model = make_model(args)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('featEmbedding.'))
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.1)
    dataset = WindowDataset(args.feature_root / f'train_gt_{args.clip_len}', len(model.pred_split['id2cls']),
                            args.window_size, args.limit_pairs)
    if not len(dataset):
        raise ValueError('no training windows found')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=padding_collate_fn)
    config = {key: str(value) if hasattr(value, '__fspath__') else value for key, value in vars(args).items()}
    path = checkpoint_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_map = -float('inf')
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('config') != config:
            raise ValueError('output checkpoint belongs to another configuration; choose a new output_dir')
        best_map = saved['mean_map']
    for epoch in range(args.num_epochs):
        logger.info('================ Epoch: %s ================', epoch + 1)
        model.train()
        losses = []
        total = min(len(loader), args.max_steps) if args.max_steps else len(loader)
        progress = tqdm(loader, total=total, desc=f'Train {epoch + 1}/{args.num_epochs}', ncols=args.progress_width)
        for step, (features, lengths, labels) in enumerate(progress):
            features = {key: value.to(args.device) for key, value in features.items()}
            labels = {key: value.to(args.device) for key, value in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(features, lengths.to(args.device), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f'{loss.item():.4f}')
            if args.max_steps and step + 1 >= args.max_steps:
                break
        scheduler.step()
        mean_loss = sum(losses) / len(losses)
        logger.info('Training loss (mean batch loss): %.6f', mean_loss)
        metrics, mean_map = evaluate_splits(model, args, logger)
        checkpoint = dict(format='stream1', stage='stage1', state_dict=model.state_dict(),
                          optimizer=optimizer.state_dict(), epoch=epoch + 1, window_size=args.window_size,
                          clip_len=args.clip_len, pred_split=model.pred_split, obj_split=model.obj_split,
                          metrics=metrics, map=[metrics['novel'][0] * 100, metrics['all'][0] * 100],
                          config=config, text={key: getattr(model, 'text_' + key).detach().cpu()
                                               for key in ('subject', 'object', 'predicate', 'objects')})
        previous_best = best_map
        best_map = save_best(checkpoint, path, mean_map, best_map)
        logger.info('Mean mAP: %.4f, Best Mean mAP: %.4f', mean_map * 100, best_map * 100)
        if mean_map > previous_best:
            logger.info('Save best checkpoint to %s', path)


if __name__ == '__main__':
    main()
