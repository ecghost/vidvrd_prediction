# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from models.relationship_multimodel.model_tuing_plus_repro_copy_new_cross_dataset_stage1_s import Model
# from models.relationship_multimodel.model_tuing_plus_repro_copy_new_cross_dataset_stage2_s import Model

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
from dataset_relaionship import DatasetRelationshipMultimodelTem, DatasetRelationshipMultimodel, padding_collate_fn_multimodel, Dataset, padding_collate_fn, padding_collate_fn_multimodel
from utils.relationship.parser_func import parse_args
from utils.relationship.utils import get_feat_types, AverageMeter, get_logger, print_results
from utils.relationship.post_process import process_pred, association, format_
from utils.relationship.video_relation_detection_openvoc import eval_relation_detection_openvoc, eval_relation_detection_openvoc_v2, eval_relation_detection_rela
import json
from utils.script import *

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
    json_tool = JsonTool()
    seed_everything(42)
    args = parse_args()
    args.batch_size=32
    # args.batch_size=32
    args.lr = 1e-4
    # args.batch_size = 1
    # args.lr = 1e-5
    task_name = "multimodel_new_stage1_output"
    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = 'rela_' + args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+str(args.temp_model)+ \
        feat_config+args.ps
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
    pred_cls_split_info = json_tool.read_json("/data/cyl/project/MM_VidVRD/project/dataset_new/anno/rela_split_and_id_cls_modify.json")
    logging.info(f"Start training\nLog path : {log_path}\nCheckpoint path : {checkpoint_folder}")

    logging.info("Preparing data from %s..."%args.dataset)
    train_dataset = DatasetRelationshipMultimodel(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=padding_collate_fn, num_workers=8)
    # SGDet
    # args.test_traj = "cls"
    # SGCls
    # args.test_traj = "gt_cls"
    args.test_traj = "gt"
    val_dataset_det = DatasetRelationshipMultimodel(args, "test")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    # # SGDet
    # args.test_traj = "cls"
    # SGCls
    # args.test_traj = "sgcls_multi"
    # val_dataset_det_sgdet = DatasetRelationshipMultimodel(args, "test")
    # val_loader_det_sgdet = DataLoader(val_dataset_det_sgdet, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    # # SGDet
    # # args.test_traj = "cls"
    # # SGCls
    # args.test_traj = "gt_cls"
    # val_dataset_det_sgcls = DatasetRelationshipMultimodel(args, "test")
    # val_loader_det_sgcls = DataLoader(val_dataset_det_sgcls, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)

    # logging.info("Loading model ...")
    model = Model(args).cuda()
    # ckpt = torch.load("/data/cyl/project/MM_VidVRD/project/code_new/output/relationship/checkpoint/rela_mm_vidvrd_bs1_lr1e-05_dim512_None_rel_mot_clip_bbox__multimodel_stage1.pth")
    # pretrained_dict = ckpt['state_dict']
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 30, 35], gamma=0.1)
    
    # total, trainable = count_parameters(model)
    # logging.info(f"Total params: {total:,}")
    # asadfsdafasdfd
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.1)

    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad_(False)
        elif "featEmbedding" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    # for name, param in model.named_parameters():
    #     print(name)
    #     if "text_encoder" in name:
    #         param.requires_grad_(True)
    #     elif "featEmbedding" in name:
    #         param.requires_grad_(False)
    #     else:
    #         param.requires_grad_(True)

    # from fvcore.nn import FlopCountAnalysis, flop_count_table

    # class ModelForFlops(torch.nn.Module):
    #     """
    #     包装一下你的模型，让 forward 只接受  
    #     (feats, seq_lens, labels)
    #     这样 fvcore 才能正常 trace。
    #     """
    #     def __init__(self, model):
    #         super().__init__()
    #         self.model = model

    #     def forward(self, feats, seq_lens, labels):
    #         return self.model(feats, seq_lens, labels)

    # # —— 从 train_loader 取一个 batch ——
    # sample_data, sample_seq_lens = next(iter(train_loader))

    # # —— 构造 labels (tensor) ——
    # labels = {
    #     'pre_label': sample_data['pre_label'].cuda(),
    #     'sbj_label': sample_data['sbj_label'].cuda(),
    #     'obj_label': sample_data['obj_label'].cuda()
    # }

    # # —— 构造 feats（只要包含 feat 或 frame 的键），全部 tensor 化 —— 
    # feats = {}
    # for k in sample_data:
    #     if "feat" in k or "frame" in k:
    #         # 如果是 numpy → tensor
    #         if isinstance(sample_data[k], (np.ndarray, np.generic)):
    #             feats[k] = torch.tensor(sample_data[k]).cuda()
    #         else:
    #             feats[k] = sample_data[k].cuda()

    # seq_lens = sample_seq_lens  # 一般是 list/int，不用放 cuda

    # # —— 包装模型 ——
    # wrapped_model = ModelForFlops(model).eval().cuda()

    # print("\n================ FLOPs Counting ================")
    # try:
    #     flops = FlopCountAnalysis(wrapped_model, (feats, seq_lens, labels))
    #     print(f"FLOPs per forward pass: {flops.total() / 1e9:.3f} GFLOPs")
    #     print(flop_count_table(flops, max_depth=3))
    # except Exception as e:
    #     print("FLOPs calculation failed:", e)

    # sadfadsfasdfasdf

    epoch_loss = AverageMeter()
    best_mmap = 0
    for epoch in range(args.start_epoch+1, args.max_epoch+1):   
        logging.info(f"=============================== Epoch : {epoch} =====================================")
        model.train()
        batch_loss = AverageMeter()
        for idx, (data, seq_lens) in enumerate(tqdm(train_loader)):
            labels = {
                'pre_label': data['pre_label'].cuda(),
                'sbj_label': data['sbj_label'].cuda(),
                'obj_label': data['obj_label'].cuda()
            }
            feats = {}
            for k in data:
                if 'feat' in k or 'frame' in k:
                    feats[k] = data[k]
            loss = model(feats, seq_lens, labels)
            # print(loss.item())
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss.update(loss.item()/args.batch_size, args.batch_size)
            epoch_loss.update(loss.item()/args.batch_size, args.batch_size)
        scheduler.step()
        logging.info(f"Avg Train Loss:  {epoch_loss.avg:.4f}")
        epoch_loss.reset()

        model.eval()
        map_list = []
        rel_tsne = {}



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
                pre_preds, sbj_preds, obj_preds, _ = model(feats, seq_lens)
                for seq_id, seq_len in enumerate(seq_lens):
                    clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                    pred_rels[vids[seq_id]].extend(association(clip_rels))
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, pred_cls_split_info=pred_cls_split_info)
        logging.info("PredCls and Novel split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # asdfasdfasdf
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
                pre_preds, sbj_preds, obj_preds, _ = model(feats, seq_lens)
                for seq_id, seq_len in enumerate(seq_lens):
                    clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                    pred_rels[vids[seq_id]].extend(association(clip_rels))
                # break
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, pred_cls_split_info=pred_cls_split_info)
        logging.info("PredCls and All split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)

        # model.tgt_split = 'novel'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in tqdm(val_loader_det_sgdet):
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_det_sgdet.id2pre, val_dataset_det_sgdet.obj2id, val_dataset_det_sgdet.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels, pred_cls_split_info=pred_cls_split_info)
        # logging.info("SGDet and Novel split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))

        # map_list.append(mean_ap*100)
        # model.tgt_split = 'all'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in tqdm(val_loader_det_sgdet):
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_det_sgdet.id2pre, val_dataset_det_sgdet.obj2id, val_dataset_det_sgdet.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, pred_cls_split_info=pred_cls_split_info)
        # logging.info("SGDet and All split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # map_list.append(mean_ap*100)


        # model.tgt_split = 'novel'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in tqdm(val_loader_det_sgcls):
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_det_sgcls.id2pre, val_dataset_det_sgcls.obj2id, val_dataset_det_sgcls.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels, pred_cls_split_info=pred_cls_split_info)
        # logging.info("SGCls and Novel split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))

        # map_list.append(mean_ap*100)
        # model.tgt_split = 'all'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in tqdm(val_loader_det_sgcls):
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_det_sgcls.id2pre, val_dataset_det_sgcls.obj2id, val_dataset_det_sgcls.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, pred_cls_split_info=pred_cls_split_info)
        # logging.info("SGCls and All split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # map_list.append(mean_ap*100)


        mmap = sum(map_list)/2
        logging.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")
        if mmap > best_mmap:
            best_mmap = mmap
            state = {
                'map': map_list,
                'epoch': epoch,
                'config': args,
                'state_dict': model.state_dict()}
            ckpt_path = join(checkpoint_folder, env_config+f"_{task_name}.pth")
            torch.save(state, ckpt_path)
            logging.info(f"Save checkpoint to {ckpt_path}")

