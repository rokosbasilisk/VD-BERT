from random import randint, shuffle, choices, choice
from random import random as rand
import math
import json
import torch
import torch.utils.data
import torch.nn.functional as F
from loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
from grid_utils import get_3d_repr

import numpy as np
import h5py
from tqdm import tqdm
import pickle
from typing import List
import copy

class IGLUDataset(torch.utils.data.Dataset):
    """ Load image-sentence pairs """

    def __init__(self,batch_size, tokenizer, data_path,s2s_data=None,is_test=False):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.data_path  = data_path
        self.batch_size = batch_size
        self.s2s_data = s2s_data
        self.is_test = is_test
        with open(self.data_path,'rb') as f:
            self.samples = pickle.load(f)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prev_utt = sample['prev_utterances']
        cur_hist = []
        for utt in prev_utt:
            utt = utt['utterance']
            if type(utt) == type(list()):
                utt = utt[0].strip("<>")
                utt = utt.replace('_',' ')
                cur_hist.append(utt)
            else:
                cur_hist.append(utt)
        ques_tokens = self.tokenizer.tokenize(cur_hist[-1]+'?')
        cur_hist = " ".join(cur_hist[:-1])
        hist_tokens = self.tokenizer.tokenize(cur_hist)
        if self.is_test == False:
            ans_tokens = self.tokenizer.tokenize(sample['next_utterance'])
        built = get_3d_repr(sample['built_config'])
        gold = get_3d_repr(sample['gold_config'])
        diff_ = gold-built
        img3d = torch.cat([built,gold,diff_])
        if self.is_test == False:
            instance = img3d,(hist_tokens,ques_tokens,ans_tokens,1.0)
        else:
            instance = img3d,(hist_tokens,ques_tokens)
        return self.s2s_data(instance)

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.samples) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.samples) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)

class Preprocess4IGLU(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={}, mask_image_regions=False, mode="s2s", vis_mask_prob=0.25,
                 region_bbox_file='', region_det_file_prefix='',pad_hist=False, finetune=False, 
                 only_mask_ans=False, float_nsp_label=False, add_boundary=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)
        self.mask_image_regions = mask_image_regions
        self.pad_hist = pad_hist
        self.finetune = finetune
        self.only_mask_ans = only_mask_ans
        self.add_boundary = add_boundary
        self.float_nsp_label = float_nsp_label

        self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]
        self.vis_mask_prob = vis_mask_prob

    def __call__(self, instance):
        img3d, iglu_example = instance # (img3d, (cur_hist,ques_tokens,ans_tokens))
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        assert isinstance(iglu_example, tuple)
        hist_tokens, ques_tokens, ans_tokens, nsp_label = iglu_example
        if len(ques_tokens) < self.max_len_hist_ques:
            if self.pad_hist:
                hist_tokens = pad_to_length(hist_tokens, self.max_len_hist_ques - len(ques_tokens))
            else:
                hist_tokens = hist_tokens[:self.max_len_hist_ques - len(ques_tokens)]
        else:
            hist_tokens = []
            ques_tokens = ques_tokens[:self.max_len_hist_ques]

        prev_tokens = hist_tokens + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
        if self.pad_hist:
            assert len(prev_tokens) == self.max_len_hist_ques + 2
        else:
            assert len(prev_tokens) <= self.max_len_hist_ques + 2

        ans_tokens = ans_tokens[:self.max_len_ans]

        tokens_b = prev_tokens + ans_tokens
        assert len(tokens_b) <= self.max_len_hist_ques + 2 + self.max_len_ans

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        assert len(tokens) <= self.max_len

        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)  # yue, do not new_segment_ids

        if self.only_mask_ans:
            effective_length = len(ans_tokens)
            start_id = len(prev_tokens) + self.len_vis_input + 2
        else:
            effective_length = len(tokens_b)
            start_id = len(tokens_a) + 2

        if self.max_pred == 0:
            input_ids = self.indexer(tokens)
            masked_ids = masked_pos = masked_weights = []
        else:
            if nsp_label == 0:
                assert tokens.count('[SEP_1]') == 1
                end_id = tokens.index('[SEP_1]')
            else:
                end_id = len(tokens)
            input_ids, masked_ids, masked_pos, masked_weights = self.conduct_mask(tokens,
                                                                                  effective_length,
                                                                                  start_id, end_id)
        input_mask, vis_masked_pos = self.get_attn_mask(tokens, len(prev_tokens))
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if self.float_nsp_label:
            nsp_label = torch.tensor(nsp_label, dtype=torch.float32)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, nsp_label, self.task_idx,
                vis_masked_pos, img3d)

    def get_attn_mask(self, tokens, prev_tokens_len):
        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

        if self.only_mask_ans:
            input_mask[:, :self.len_vis_input + 2 + prev_tokens_len].fill_(1)
            second_st, second_end = self.len_vis_input + 2 + prev_tokens_len, len(tokens)
        else:
            input_mask[:, :self.len_vis_input + 2].fill_(1)
            second_st, second_end = self.len_vis_input + 2, len(tokens)
        input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])

        if self.pad_hist:
            padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
            if len(padded_pos) > 0:
                assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
                input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)

        if self.mask_image_regions:
            vis_masked_pos = np.random.choice(self.len_vis_input,
                                              int(self.len_vis_input * self.vis_mask_prob),
                                              replace=False) + 1  # +1 for [CLS], always of the same length
        else:
            vis_masked_pos = []

        if self.mask_image_regions:
            input_mask[:, vis_masked_pos].fill_(0)  # block the masked visual feature
        return input_mask, vis_masked_pos
    def conduct_mask(self, tokens, effective_length, start_id, end_id):
        # For masked Language Models
        cand_pos = []
        special_pos = set()

        n_pred = min(self.max_pred, max(
            1, int(round(effective_length * self.mask_prob))))

        # candidate positions of masked tokens
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= start_id) and (tk != '[CLS]') and (tk != '[PAD]') and (i < end_id):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)
        masked_pos = cand_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if self.finetune:
                tokens[pos] = '[MASK]'
                continue
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        assert len(masked_ids) == len(masked_pos) == len(masked_weights) == self.max_pred, \
            "[masked] id: %d, pos: %d, weights: %d" % (len(masked_ids), len(masked_pos), len(masked_weights))

        return input_ids, masked_ids, masked_pos, masked_weights


class Preprocess4IGLUGen(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={},mode="s2s",pad_hist=False, inc_full_hist=False):
        super().__init__()
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)

        self.mode = mode
        self.pad_hist = pad_hist
        self.inc_full_hist = inc_full_hist

        self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]

    def __call__(self, instance):
        img3d, (cur_hist, ques_tokens)= instance
        cur_hist = cur_hist[:20]
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        cur_hist = pad_to_length(cur_hist, self.max_len_hist_ques - len(ques_tokens))

        prev_tokens = cur_hist + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']


        tokens_b = pad_to_length(prev_tokens, self.max_len_hist_ques + 2)

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (self.max_len_hist_ques + 2 + self.max_len_ans+1)
        input_ids = self.indexer(tokens)

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :self.len_vis_input + 2 + self.max_len_hist_ques + 2].fill_(1)
        second_st, second_end = self.len_vis_input + 2 + self.max_len_hist_ques + 2, self.max_len
        input_mask[second_st:second_end, second_st:second_end].copy_(
        self._tril_matrix[:second_end - second_st, :second_end - second_st])

        padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
        if len(padded_pos) > 0:
            assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
            input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)

            # Need to align them for decoding, use position ids to identify
        position_ids = []
        for idx in range(self.max_len):
            if idx < self.len_vis_input + 2 + len(prev_tokens):
                position_ids.append(idx)
            elif idx >= self.len_vis_input + 2 + self.max_len_hist_ques + 2:
                shift_idx = idx - (self.max_len_hist_ques + 2 - len(prev_tokens))
                position_ids.append(shift_idx)
            else:
                position_ids.append(0)

        return (input_ids, segment_ids, position_ids,input_mask, self.task_idx, img3d)

