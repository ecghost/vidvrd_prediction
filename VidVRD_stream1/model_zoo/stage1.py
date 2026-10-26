import json

import torch
from torch import nn
from torch.nn import functional as F


class SpatialDecoder(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        self.role_embeddings = nn.Embedding(4, dim)
        self.relEmb = nn.Linear(dim, dim, bias=False)
        self.objEmb = nn.Linear(dim, dim, bias=False)
        self.boxEmb = nn.Sequential(nn.Linear(24, dim), nn.ReLU(), nn.Linear(dim, dim, bias=False))
        self.posEmb = nn.Sequential(nn.Linear(84, dim), nn.ReLU(), nn.Linear(dim, dim, bias=False))

    def forward(self, x_img, x_box, x_pos):
        roles = F.normalize(self.role_embeddings(torch.arange(4, device=x_img.device)), dim=-1)
        x_img = x_img + F.normalize(self.boxEmb(x_box), dim=-1) + roles
        x_img = F.normalize(self.transformer(x_img), dim=-1)
        objects = F.normalize(self.objEmb(x_img[:, :2]), dim=-1)
        positions = F.normalize(self.posEmb(x_pos), dim=-1).unsqueeze(1)
        return F.normalize(x_img + positions, dim=-1), objects[:, 0], objects[:, 1]


class TemporalDecoder(nn.Module):
    def __init__(self, dim=768, heads=8, capacity=40):
        super().__init__()
        self.intHead = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(),
                                     nn.Linear(dim, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))
        self.transformer = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        self.pos_embeddings = nn.Embedding(capacity, dim)
        self.relEmb = nn.Linear(dim, dim, bias=False)

    def forward(self, tokens, seq_lens):
        batch, length, roles, dim = tokens.shape
        if length > self.pos_embeddings.num_embeddings:
            raise ValueError('window_size exceeds temporal position capacity')
        seq_lens = torch.as_tensor(seq_lens, device=tokens.device)
        if (seq_lens < 1).any() or (seq_lens > length).any():
            raise ValueError('invalid seq_len')
        positions = torch.arange(length, device=tokens.device)
        padding = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
        hidden = tokens.permute(0, 2, 1, 3).reshape(batch * roles, length, dim)
        hidden = hidden.masked_fill(padding.repeat_interleave(roles, 0).unsqueeze(-1), 0)
        hidden = hidden + F.normalize(self.pos_embeddings(positions), dim=-1)
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        hidden = self.transformer(hidden, src_mask=causal,
                                  src_key_padding_mask=padding.repeat_interleave(roles, 0))
        hidden = hidden.reshape(batch, roles, length, dim).permute(0, 2, 1, 3)
        return F.normalize(self.relEmb(hidden), dim=-1), self.intHead(hidden.flatten(2)).squeeze(-1)


def valid_mean(values, seq_lens):
    mask = torch.arange(values.shape[1], device=values.device)[None] < seq_lens[:, None]
    return values.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / seq_lens.unsqueeze(-1)


class FeatEmbedding(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.spatial_decoder = SpatialDecoder(dim, heads)
        self.temporal_decoder = TemporalDecoder(dim, heads)

    def encode_spatial(self, inputs):
        batch, length = inputs['clip_feat'].shape[:2]
        relation, subject, obj = self.spatial_decoder(
            inputs['clip_feat'].reshape(batch * length, 4, -1),
            inputs['bbox_feat'].reshape(batch * length, 4, 24),
            torch.cat([inputs['rel_feat'], inputs['mot_feat']], -1).reshape(batch * length, 84))
        return (relation.reshape(batch, length, 4, -1),
                subject.reshape(batch, length, -1), obj.reshape(batch, length, -1))


def focal(logits, targets):
    logpt = -F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (-((1 - logpt.exp()) ** 2) * logpt).mean()


class Model(nn.Module):
    def __init__(self, text, pred_split, obj_split, window_size=4, src_split='base',
                 tgt_split='all', obj_loss_weight=0.2, int_loss_weight=1., dim=768, heads=8):
        super().__init__()
        if not 1 <= window_size <= 40:
            raise ValueError('window_size must be in [1,40]')
        self.window_size = window_size
        self.src_split, self.tgt_split = src_split, tgt_split
        self.pred_split, self.obj_split = pred_split, obj_split
        self.obj_loss_weight, self.int_loss_weight = obj_loss_weight, int_loss_weight
        self.temperature = 0.01
        self.featEmbedding = FeatEmbedding(dim, heads)
        for name in ('subject', 'object', 'predicate', 'objects'):
            self.register_buffer('text_' + name, F.normalize(text[name].float(), dim=-1))
        if any(text[name].ndim != 2 or text[name].shape[1] != dim for name in text):
            raise ValueError('text vectors must have shape [classes, dim]')
        if any(len(text[name]) != len(pred_split['id2cls']) for name in ('subject', 'object', 'predicate')):
            raise ValueError('predicate text count does not match split')
        if len(text['objects']) != len(obj_split['id2cls']):
            raise ValueError('object text count does not match split')

    def ids(self, info, split):
        return [index for index in range(len(info['id2cls']))
                if split == 'all' or info['cls2split'][info['id2cls'][str(index)]] == split]

    def encode_spatial(self, inputs):
        return self.featEmbedding.encode_spatial(inputs)

    def classify_current(self, spatial, seq_lens, split):
        relation, subject, obj = spatial
        seq_lens = torch.as_tensor(seq_lens, device=relation.device, dtype=torch.long)
        if relation.shape[1] > self.window_size:
            raise ValueError('input exceeds configured window_size')
        temporal, interaction = self.featEmbedding.temporal_decoder(relation, seq_lens)
        batch_ids = torch.arange(len(seq_lens), device=relation.device)
        current = F.normalize(temporal[batch_ids, seq_lens - 1].flatten(1) / 2, dim=-1)
        pred_ids, obj_ids = self.ids(self.pred_split, split), self.ids(self.obj_split, split)
        if not pred_ids or not obj_ids:
            raise ValueError('selected split has no classes')
        text = torch.cat([self.text_subject[pred_ids], self.text_object[pred_ids],
                          self.text_predicate[pred_ids], self.text_predicate[pred_ids]], -1) / 2
        pre_logits = current @ text.T / self.temperature
        sbj_logits = F.normalize(valid_mean(subject, seq_lens), dim=-1) @ self.text_objects[obj_ids].T / self.temperature
        obj_logits = F.normalize(valid_mean(obj, seq_lens), dim=-1) @ self.text_objects[obj_ids].T / self.temperature
        return pre_logits, interaction[batch_ids, seq_lens - 1], sbj_logits, obj_logits

    def forward(self, inputs, seq_lens, labels=None):
        split = self.src_split if labels is not None else self.tgt_split
        logits = self.classify_current(self.encode_spatial(inputs), seq_lens, split)
        if labels is None:
            return self.scores(logits)
        pre_logits, int_logits, sbj_logits, obj_logits = logits
        pred_ids = self.ids(self.pred_split, split)
        object_ids = self.ids(self.obj_split, split)
        mapping = torch.full((len(self.obj_split['id2cls']),), -1, device=pre_logits.device, dtype=torch.long)
        mapping[object_ids] = torch.arange(len(object_ids), device=mapping.device)
        def object_loss(values, targets):
            converted = torch.full_like(targets, -1)
            valid = targets >= 0
            converted[valid] = mapping[targets[valid]]
            if not (converted >= 0).any():
                return values.sum() * 0
            return F.cross_entropy(values, converted, ignore_index=-1)
        pre_labels = labels['pre_label'].float()
        rel_loss = torch.stack([focal(pre_logits[index], pre_labels[index, pred_ids])
                                for index in range(len(pre_logits))]).sum()
        int_targets = (pre_labels.sum(-1) > 0).float()
        int_loss = torch.stack([focal(int_logits[index], int_targets[index])
                                for index in range(len(pre_logits))]).sum()
        obj_loss = object_loss(sbj_logits, labels['sbj_label']) + object_loss(obj_logits, labels['obj_label'])
        return rel_loss + self.int_loss_weight * int_loss + self.obj_loss_weight * obj_loss

    def scores(self, logits):
        pre_logits, interaction, _, _ = logits
        scores = pre_logits.new_zeros(len(pre_logits), len(self.pred_split['id2cls']))
        scores[:, self.ids(self.pred_split, self.tgt_split)] = pre_logits.sigmoid() * interaction.sigmoid()[:, None]
        return scores


def load_checkpoint(model, path, allow_partial=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if checkpoint.get('stage', 'stage1') != 'stage1' or any('prompt_learner' in key or 'meta_net' in key or 'pre_classifier.' in key for key in state):
        raise ValueError('Stage 2 checkpoint is not supported')
    own = model.state_dict()
    loaded, extra, mismatched = {}, [], []
    for key, value in state.items():
        key = key.removeprefix('module.')
        if key not in own:
            extra.append(key)
        elif value.shape != own[key].shape:
            mismatched.append(key)
        else:
            loaded[key] = value
    missing = sorted(set(own) - set(loaded))
    report = dict(loaded=sorted(loaded), missing=missing, extra=extra, shape_incompatible=mismatched)
    print(json.dumps(report, indent=2))
    critical = [key for key in missing if key.startswith('featEmbedding.')]
    if critical and not allow_partial:
        raise ValueError('missing/incompatible spatial or temporal parameters; use --allow_partial_init explicitly')
    model.load_state_dict(loaded, strict=False)
    return checkpoint
