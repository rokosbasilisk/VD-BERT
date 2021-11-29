"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import json
import argparse
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import copy
import time

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForVisDialGen,BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from loader_utils import batch_list_to_batch_tensors
from seq2seq_loader_iglu import Preprocess4IGLUGen, IGLUDataset

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="The input data file name.")
    parser.add_argument("--neg_num", default=0, type=int)
    parser.add_argument("--only_qa", default=0, type=int)
    parser.add_argument("--no_h0", default=0, type=int)
    parser.add_argument("--no_vision", default=0, type=int)
    parser.add_argument("--sub_sample", default=0, type=int)
    parser.add_argument("--adaptive_weight", default=0, type=int)
    parser.add_argument("--multiple_neg", default=0, type=int)
    parser.add_argument("--inc_gt_rel", default=0, type=int)
    parser.add_argument("--inc_full_hist", default=0, type=int)
    parser.add_argument("--just_for_pretrain", default=0, type=int)
    parser.add_argument("--add_boundary", default=0, type=int)
    parser.add_argument("--add_attn_fuse", default=0, type=int)
    parser.add_argument("--pad_hist", default=0, type=int)
    parser.add_argument("--only_mask_ans", default=0, type=int)
    parser.add_argument("--visdial_v", default='1.0', choices=['1.0'], type=str)
    parser.add_argument("--loss_type", default='mlm', choices=['mlm'], type=str)

    parser.add_argument('--len_vis_input', type=int, default=18)
    parser.add_argument('--max_len_ans', type=int, default=10)
    parser.add_argument('--max_len_hist_ques', type=int, default=40)

    parser.add_argument("--finetune", default=0, type=int)
    parser.add_argument('--tasks', default='img2txt',
                        help='img2txt | vqa2| ctrl2 | visdial | visdial_short_hist | visdial_nsp')

    # General
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--global_rank",
                        type=int,
                        default=-1,
                        help="global_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=3,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=4, type=int,  # yue should be 4
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    # Others for VLP
    parser.add_argument('--enable_visdom', action='store_true')
    parser.add_argument('--visdom_port', type=int, default=8888)
    # parser.add_argument('--resnet_model', type=str, default='imagenet_weights/resnet101.pth')
    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'restval'])

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='file://[PT_OUTPUT_DIR]/nonexistent_file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--file_valid_jpgs', default='/mnt/dat/COCO/annotations/coco_valid_jpgs.json', type=str)
    parser.add_argument('--sche_mode', default='warmup_linear', type=str,
                        help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=500, type=int)
    parser.add_argument('--vis_mask_prob', default=0, type=float)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)

    parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq).")
    parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
    parser.add_argument('--l2r_prob', default=0, type=float,
                        help="Percentage of examples that are unidirectional (left-to-right) LM.")
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5',
                        type=str)
    parser.add_argument('--region_det_file_prefix',
                        default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--scst', action='store_true',
                        help='Self-critical sequence training')

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(d) for d in range(args.world_size)])

    print('Arguments: %s' % (' '.join(sys.argv[:])))
    return args


def main():
    args = process_args()
    # Input format: [CLS] img [SEP] hist [SEP_0] ques [SEP_1] ans [SEP]
    args.max_seq_length = args.len_vis_input + 2 + args.max_len_hist_ques + 2 + args.max_len_ans + 1

    args.mask_image_regions = (args.vis_mask_prob > 0)  # whether to mask out image regions

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()


    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
        )
    
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    
    s2s_data = Preprocess4IGLUGen(args.max_pred, args.mask_prob,
            list(tokenizer.vocab.keys()),
            tokenizer.convert_tokens_to_ids, args.max_seq_length,
            new_segment_ids=args.new_segment_ids,
            truncate_config={'len_vis_input': args.len_vis_input,
                             'max_len_hist_ques': args.max_len_hist_ques,
                             'max_len_ans': args.max_len_ans},
            mode="s2s",pad_hist=args.pad_hist)


    test_dataset = IGLUDataset(args.train_batch_size, data_tokenizer, args.data_path,s2s_data)



    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 if args.tasks == 'img2txt' else 0
    mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[PAD]"])  # index in BERT vocab: 103, 102, 0

    model_recover = torch.load(args.model_recover_path)

    model = BertForVisDialGen.from_pretrained(
            args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding,mask_word_id=mask_word_id,
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, visdial_v=args.visdial_v
            )
   
    del model_recover
    torch.cuda.empty_cache()
    model.to(device)

    torch.cuda.empty_cache()
    #input_ids, segment_ids, position_ids,input_mask,task_idx,img3d
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,collate_fn=batch_list_to_batch_tensors)
    batch = next(iter(test_dataloader))
    #batch.to(device)

    input_ids, segment_ids, position_ids,input_mask,task_idx,img3d = batch
    img3d = img3d.to(device)
    input_ids = input_ids.to(device)
    segment_ids = segment_ids.to(device)
    position_ids = position_ids.to(device)
    input_mask = input_mask.to(device)
    task_idx = task_idx.to(device)

    predictions  = model.forward(img3d,input_ids,segment_ids, position_ids, input_mask)
    print(predictions)



if __name__ == "__main__":
    main()
