import json

import torch
from torch.utils.data import DataLoader

from dataset import WindowDataset, padding_collate_fn
from runtime import make_model, replay
from utils.parser_func import parse_args


def main():
    args = parse_args()
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
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    for epoch in range(args.num_epochs):
        model.train()
        losses = []
        for step, (features, lengths, labels) in enumerate(loader):
            features = {key: value.to(args.device) for key, value in features.items()}
            labels = {key: value.to(args.device) for key, value in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(features, lengths.to(args.device), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.max_steps and step + 1 >= args.max_steps:
                break
        scheduler.step()
        checkpoint = dict(format='stream1', stage='stage1', state_dict=model.state_dict(),
                          optimizer=optimizer.state_dict(), epoch=epoch + 1, window_size=args.window_size,
                          clip_len=args.clip_len, pred_split=model.pred_split, obj_split=model.obj_split,
                          config=config, text={key: getattr(model, 'text_' + key).detach().cpu()
                                               for key in ('subject', 'object', 'predicate', 'objects')})
        torch.save(checkpoint, args.output_dir / f'epoch_{epoch + 1}.pth')
        mean_loss = sum(losses) / len(losses)
        print(f'epoch={epoch + 1} mean_batch_loss={mean_loss:.6f}')
        with open(args.output_dir / 'train.jsonl', 'a') as handle:
            handle.write(json.dumps(dict(epoch=epoch + 1, mean_batch_loss=mean_loss, steps=len(losses))) + '\n')
        replay(model, args, f'epoch_{epoch + 1}')


if __name__ == '__main__':
    main()
