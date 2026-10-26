import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def build_text_vectors(pred_info, obj_info, weights, device):
    if not weights.is_file():
        raise ValueError('text_model_weights must be a local CLIP checkpoint; downloads are disabled')
    source = Path(__file__).resolve().parent.parent / 'VidVRD_predication'
    sys.path.insert(0, str(source))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from clip import clip
    finally:
        sys.path.remove(str(source))
        sys.dont_write_bytecode = previous_bytecode
    model, _ = clip.load(str(weights), device=device)
    model.eval()
    predicates = [pred_info['id2cls'][str(index)].replace('_', ' ') for index in range(len(pred_info['id2cls']))]
    objects = [obj_info['id2cls'][str(index)].replace('_', ' ') for index in range(len(obj_info['id2cls']))]
    templates = {
        'subject': [f'An image of a person or object {name} something.' for name in predicates],
        'object': [f'An image of something {name} a person or object.' for name in predicates],
        'predicate': [f'An image of the visual relation {name} between two entities.' for name in predicates],
        'objects': [f'An image of {name}.' for name in objects]}
    with torch.no_grad():
        return {key: F.normalize(model.encode_text(clip.tokenize(prompts).to(device)).float(), dim=-1).cpu()
                for key, prompts in templates.items()}
