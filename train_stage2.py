# from model_zoo.stage2 import Model
# from model_zoo.stage1 import Model
from model_zoo.model_stage2 import Model

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
    # # args.batch_size=32
    # args.batch_size=1
    # args.lr = 1e-4
    # # args.lr = 1e-3
    # args.obj_loss_weight=0.2

    task_name = "stage2"
    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = 'rela_' + args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+str(args.temp_model)+ \
        feat_config+args.ps+ '_____'
    output_dir = f"{args.output_path}/log"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_folder = os.path.join(args.output_path, "checkpoint")
    log_path = f"{output_dir}/{env_config}_{task_name}.log"
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    pred_cls_split_info = json.load(open("/data/cyl/dataset/vidvrd/data/openvoc_pred_class_spilt_info.json", 'r'))
    logging.info(f"Start training\nLog path : {log_path}\nCheckpoint path : {checkpoint_folder}")

    logging.info("Preparing data from %s..."%args.dataset)
    train_dataset = Dataset(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=padding_collate_fn, num_workers=8)
    args.test_traj = "gt"
    val_dataset_det = Dataset(args, "test")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    
    model = Model(args).cuda()
    ckpt = torch.load("/data/wyq/VidVRD-baseline/scripts/vidvrd_ckpt/baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth")
    # ckpt = torch.load("/data/cyl/project/EOV-MMP-VidVRD/VidVRD_predication/output/checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth")
    print(ckpt['map'])
    pretrained_dict = ckpt['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25], gamma=0.1)
    
    for name, param in model.named_parameters():
        # print(name)
        if "text_encoder" in name:
            param.requires_grad_(False)
        elif "featEmbedding" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    epoch_loss = AverageMeter()
    best_mmap = 0
    for epoch in range(args.start_epoch+1, args.max_epoch+1):   
        # logging.info(f"=============================== Epoch : {epoch} ====================================")
        # model.train()
        # batch_loss = AverageMeter()
        # for idx, (data, seq_lens) in enumerate(tqdm(train_loader)):
        #     labels = {
        #         'pre_label': data['pre_label'].cuda(),
        #         'sbj_label': data['sbj_label'].cuda(),
        #         'obj_label': data['obj_label'].cuda()
        #     }
        #     feats = {}
        #     for k in data:
        #         if 'feat' in k:
        #             feats[k] = data[k]
        #     loss = model(feats, seq_lens, labels)
        #     # print(loss.item())
        #     loss.requires_grad_(True)
        #     loss.backward()
            
        #     if ((idx+1)%32)==0:
        #         optimizer.step()
        #         optimizer.zero_grad()
        #     # optimizer.step()
        #     # optimizer.zero_grad()
        #     # optimizer.step()
        #     # optimizer.zero_grad()
        #     batch_loss.update(loss.item()/args.batch_size, args.batch_size)
        #     epoch_loss.update(loss.item()/args.batch_size, args.batch_size)
        # scheduler.step()
        # logging.info(f"Avg Train Loss:  {epoch_loss.avg:.4f}")
        # epoch_loss.reset()

        model.eval()
        map_list = []
        rel_tsne = {}



        model.tgt_split = 'novel'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data, seq_lens in tqdm(val_loader_det):
                vids = data['vid']
                pair_data = data['pair_data']
                feats = {}
                for k in data:
                    if k not in ['vid', 'pair_data']:
                        feats[k] = data[k]
                pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
                for seq_id, seq_len in enumerate(seq_lens):
                    clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                    pred_rels[vids[seq_id]].extend(association(clip_rels))
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels)
        logging.info("PredCls and Novel split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)


        model.tgt_split = 'all'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data, seq_lens in tqdm(val_loader_det):
                vids = data['vid']
                pair_data = data['pair_data']
                feats = {}
                for k in data:
                    if k not in ['vid', 'pair_data']:
                        feats[k] = data[k]
                pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
                for seq_id, seq_len in enumerate(seq_lens):
                    clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                    pred_rels[vids[seq_id]].extend(association(clip_rels))
                # break
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels)
        logging.info("PredCls and All split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)

        with open("result.json", "w") as f:
            json.dump(pred_rels, f)


        # mmap = sum(map_list)/2
        # logging.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")
        # if mmap > best_mmap:
        #     best_mmap = mmap
        #     state = {
        #         'map': map_list,
        #         'epoch': epoch,
        #         'config': args,
        #         'state_dict': model.state_dict()}
        #     ckpt_path = join(checkpoint_folder, env_config+f"_{task_name}.pth")
        #     torch.save(state, ckpt_path)
        #     logging.info(f"Save checkpoint to {ckpt_path}")
