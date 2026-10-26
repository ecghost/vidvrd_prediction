import numpy as np
from collections import defaultdict
from os.path import join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import visdom
import json

from dataset_new import Dataset_new
from dataset import Dataset, padding_collate_fn
from utils.parser_func import parse_args
from arguments import get_args_parser
from utils.video_relation_detection import evaluate
from utils.utils import get_feat_types, AverageMeter, get_logger, print_results
from utils.post_process import process_pred, association, format_
from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc
import multiprocessing as mp
from methods import build_model
from end2end_model import End2End_Model
from model import Classifier
import warnings
from os.path import join
warnings.filterwarnings("ignore")
# from deep_sort_app import run
from AFLink.AppFreeLink import *


def seed_everything(seed = 3407):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(3407)
    args = parse_args()

    from VidVRD_predication.model_zoo.stage2 import Model

    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = 'baseline_fbce_' +\
        args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+str(args.temp_model)+ \
        feat_config+args.ps
    # vis = visdom.Visdom(env=env_config)

    # logger = get_logger(join('.','','project/end2end',env_config+'_train.log'))
    # logger.info('Experiment Config: {}'.format(args))

    # logger.info("Preparing data from %s..."%args.dataset)
    train_dataset = Dataset_new(args, "train")
    item = train_dataset[0]
    print(item['patch_proj'].shape, item['global_proj'].shape)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # args.test_traj = 'openmeta'
    # args.dataset = 'VidOR'
    val_dataset_det = Dataset_new(args, "val")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size, shuffle=False)
    # ckpt = torch.load("/media/sda1/wyq/VidVRD-baseline/VidVRD-baseline/dataset/vidvrd/model/wyq/baseline_fbce_vidvrd_bs32_lr0.001_drop0.5_dim512_none_rel_mot_clip_bbox_openvoc_repro_paper.pth")
    # ckpt = torch.load("/media/sda1/wyq/VidVRD-baseline/VidVRD-baseline/dataset/vidvrd/model/wyq/baseline_fbce_vidvrd_bs1_lr0.001_drop0.5_dim512_none_rel_mot_clip_bbox_2stage_ours_8_8_2.pth")
    # ckpt = torch.load('/media/sda1/wyq/VidVRD-baseline/VidVRD-baseline/dataset/vidvrd/model/wyq/baseline_fbce_vidvrd_bs32_lr0.001_drop0.5_dim512_none_rel_mot_clip_bbox_stage1_new_B16_f.pth')
    
    object_detection_model, object_detection_criterion, object_detection_postprocessors = build_model(args)
    checkpoint_path = '/data/wyq/Source_VidVRD/output_e2e_L336/checkpoint_vidvrd0059_new_1e-5.pth'
    checkpoint = torch.load(checkpoint_path)
    object_detection_model.load_state_dict(checkpoint['model'])

    relationship_classification_model = Model(args).cuda()

    ckpt = torch.load('/data/wyq/VidVRD-baseline/scripts/vidvrd_ckpt/baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth')
    pretrained_dict = ckpt['state_dict']
    model_dict = relationship_classification_model.state_dict()
    # pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    relationship_classification_model.load_state_dict(model_dict)

    object_classifer = Classifier(args).cuda()
    checkpoint_path = '/data/wyq/object_classification_vidvrd/ckpt/vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth'
    checkpoint = torch.load(checkpoint_path)
    object_classifer.load_state_dict(checkpoint['state_dict'])

    model = End2End_Model(args, object_detection_model, object_classifer, relationship_classification_model,
                          object_detection_criterion, object_detection_postprocessors).cuda()

    # checkpoint_path = '/data/cyl/project/end2end/VidVRD/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base.pth'
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['state_dict'])

    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad_(False)
        elif 'backbone' in name:
            param.requires_grad_(False)
        elif 'pre_classifier' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    sort_model = PostLinker()
    sort_model.load_state_dict(torch.load(args.path_AFLink))
    epoch_loss = AverageMeter()
    best_mmap = 0
    for epoch in range(args.start_epoch+1, args.max_epoch+1):      
 
        model.train()
        # logger.info("Training data from %s..."%args.dataset)
        batch_loss = AverageMeter()
        a = 1
        for idx, data in enumerate(tqdm(train_loader)):
            # print(a)
            # a+=1
            print("11111111111111111111111111111111")
            print(data['patch_proj'].shape)
            model(data, sort_model)
        #     # if ((idx+1)%32)==0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     # if idx == len(train_loader) - 1:
        #     #     optimizer.step()
        #     #     optimizer.zero_grad()

        #     # batch_loss.update(loss.item()/args.batch_size, args.batch_size)
        #     # epoch_loss.update(loss.item()/args.batch_size, args.batch_size)
        #     # if (idx+1) % args.print_freq == 0:
        #     #     vis.line([batch_loss.avg], [(epoch-1)*len(train_loader) + idx+1], win="Batch Loss", update='append', opts=dict(title='batch_loss'))
        #     #     batch_loss.reset()
        # scheduler.step()

        model.eval()
        map_list = []
        
        # import json
        # split = json.load(open(join('..', 'dataset', args.dataset, 'data', 'predicate_split.json'), 'r'))
        # model.tgt_split = 'all'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in tqdm(val_loader_det):
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # results = evaluate(val_dataset_det.gt_rels, pred_rels, split['full'])
        # logger.info('==============Evaluation for Conventional Detection on Det Traj==============')
        # print_results(logger, results, verbose=False)

        # model.modelC.tgt_split = 'all'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data in tqdm(val_loader_det):
        #         final_results = model(data,sort_model)
        #         for final_result in final_results:
        #             pre_preds=final_result['pre_preds'] 
        #             sbj_preds=final_result['sbj_preds'] 
        #             obj_preds=final_result['obj_preds'] 
        #             seq_lens=final_result['seq_lens'] 
        #             vids=final_result['video_name'] 
        #             pair_data = final_result['pair_data']
        #             for seq_id, seq_len in enumerate(seq_lens):
        #                 clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #                 pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        

        

        # # # split = json.load(open(join('..', 'dataset', args.dataset, 'data', 'predicate_split.json'), 'r'))       
        # # # results = evaluate(val_dataset_det.gt_rels, pred_rels, split['full'])

        # # # print_results(logger, results, verbose=True)

        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, rt_hit_infos=True)
        # with open("./vidvrd_sample_case_end2end_sgcls_all.json","w") as f:
        #     json.dump(pred_rels,f)
        # logger.info("SGDet and All split     | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # map_list.append(mean_ap*100)
    
        # with open('./traj.json', 'w') as file:
        #     json.dump(model.traj, file, indent=4)
        
        model.modelC.tgt_split = 'novel'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data in tqdm(val_loader_det):
                final_results = model(data,sort_model)
                for final_result in final_results:
                    pre_preds=final_result['pre_preds'] 
                    sbj_preds=final_result['sbj_preds'] 
                    obj_preds=final_result['obj_preds'] 
                    seq_lens=final_result['seq_lens'] 
                    vids=final_result['video_name'] 
                    pair_data = final_result['pair_data']
                    for seq_id, seq_len in enumerate(seq_lens):
                        clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                        pred_rels[vids[seq_id]].extend(association(clip_rels))
        for vid in pred_rels:
            pred_rels[vid] = format_(args, pred_rels[vid])
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels, rt_hit_infos = True)
        with open("./vidvrd_sample_case_end2end_sgdet_novel.json","w") as f:
            json.dump(pred_rels,f)
        # logger.info("SGDet and Novel split   | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # map_list.append(mean_ap*100)
        
        # model.tgt_split = 'all'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in val_loader_gt:
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_gt.id2pre, val_dataset_gt.obj2id, val_dataset_gt.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels)
        # logger.info("PredCls and All split   | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # map_list.append(mean_ap*100)

        # model.tgt_split = 'novel'
        # pred_rels = defaultdict(list)
        # with torch.no_grad():
        #     for data, seq_lens in val_loader_gt:
        #         vids = data['vid']
        #         pair_data = data['pair_data']
        #         feats = {}
        #         for k in data:
        #             if k not in ['vid', 'pair_data']:
        #                 feats[k] = data[k]
        #         pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
        #         for seq_id, seq_len in enumerate(seq_lens):
        #             clip_rels = process_pred(args, val_dataset_gt.id2pre, val_dataset_gt.obj2id, val_dataset_gt.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
        #             pred_rels[vids[seq_id]].extend(association(clip_rels))
        # for vid in pred_rels:
        #     pred_rels[vid] = format_(args, pred_rels[vid])
        # mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels)
        # logger.info("PredCls and Novel split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        # map_list.append(mean_ap*100)

        # mmap = sum(map_list)/2

        # logger.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")
        # if mmap > best_mmap:
        #     best_mmap = mmap
        #     state = {
        #         'map': map_list,
        #         'epoch': epoch,
        #         'config': args,
        #         'state_dict': model.state_dict()}
        #     ckpt_path = join('/data/cyl/project/end2end/VidVRD/ckpt', env_config+".pth")
        #     torch.save(state, ckpt_path)
    
    print("================================Final Results======================================")
    # logger.info('Best Epoch: {}, mAP List: {}'.format(state['epoch'], str(state['map'])))



