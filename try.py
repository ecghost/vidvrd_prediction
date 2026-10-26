from model_zoo.stage2 import Model
# from model_zoo.stage1 import Model

import numpy as np
from collections import defaultdict
from os.path import join
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import visdom
from dataset import Dataset, padding_collate_fn
from utils.parser_func import parse_args
from utils.utils import get_feat_types, AverageMeter, get_logger, print_results
from utils.post_process import process_pred, association, format_
from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc
import json

def seed_everything(seed = 3047):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    seed_everything(3407)
    args = parse_args()
    # args.batch_size=32
    args.batch_size=1
    args.lr = 1e-4
    # args.lr = 1e-3
    args.obj_loss_weight=0.2

    pred_cls_split_info = json.load(open("/data/cyl/dataset/vidvrd/data/openvoc_pred_class_spilt_info.json", 'r'))
    
    model = Model(args).cuda()
    ckpt = torch.load("/data/wyq/VidVRD-baseline/scripts/vidvrd_ckpt/baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth")
    # ckpt = torch.load("/data/cyl/project/EOV-MMP-VidVRD/VidVRD_predication/output/checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth")
    print(ckpt['config'])
    pretrained_dict = ckpt['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25], gamma=0.1)
    