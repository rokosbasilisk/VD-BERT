# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Changes have been made over the original file
# https://github.com/huggingface/pytorch-transformers/blob/v0.4.0/pytorch_pretrained_bert/modeling.py

"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
import pickle

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from more_itertools import split_after
from .file_utils import cached_path
from .loss import LabelSmoothingLoss
from torch.nn.utils.rnn import pad_sequence
from .rank_loss import *

# import visdom

logger = logging.getLogger(__name__)
# vis = visdom.Visdom(port=8888, env='vlp')

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

## gridnet
class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        self.conv1 = nn.Conv3d(18,24,1)
        self.conv2 = nn.Conv3d(24,36, 1)
        self.conv3 = nn.Conv3d(36,48,1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(52272,768*18)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        output = output.view(-1,18,768)
        return output

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 label_smoothing=None):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.label_smoothing = label_smoothing
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, vis_feats, input_ids, token_type_ids=None, position_ids=None, vis_input=True,
                len_vis_input=49):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # modified
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if vis_input and len_vis_input != 0:
            words_embeddings = torch.cat((words_embeddings[:, :1], vis_feats,words_embeddings[:, len_vis_input + 1:]), dim=1)
            # assert len_vis_input == 100, 'only support region attn!'
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
                      :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, prev_embedding=None, prev_encoded_layers=None,
                output_all_encoded_layers=True):
        assert (prev_embedding is None) == (prev_encoded_layers is None), \
            "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        if pooled_output is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        if ('config_path' in kwargs) and kwargs['config_path']:
            config_file = kwargs['config_path']
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # define new relax_projection
        if ('task_idx' in kwargs) and kwargs['task_idx']:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # use fp32 for embeddings
        if ('fp32_embedding' in kwargs) and kwargs['fp32_embedding']:
            config.fp32_embedding = kwargs['fp32_embedding']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        if 'drop_prob' in kwargs:
            print('setting the new dropout rate!', kwargs['drop_prob'])
            config.attention_probs_dropout_prob = kwargs['drop_prob']
            config.hidden_dropout_prob = kwargs['drop_prob']

        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in (
                'config_path', 'type_vocab_size', 'relax_projection', 'task_idx', 'max_position_embeddings',
                'fp32_embedding',
                'label_smoothing', 'drop_prob'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info(
                "config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                    config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(
                state_dict[_k].data = state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1]).data
                if config.type_vocab_size >= 6:
                    # L2R
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                    # R2L
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                    # S2S
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info(
                "config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                    config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                state_dict[_k].data = state_dict[_k].data.resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start + chunk_size,
                    :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        # initialize relax projection
        _k = 'cls.predictions.transform.dense.weight'
        n_config_relax = 1 if (config.relax_projection <
                               1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax * config.hidden_size != state_dict[_k].shape[0]):
            logger.info(
                "n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(
                    n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = state_dict[_k].shape[0] // config.hidden_size
            assert (n_state_relax == 1) != (n_config_relax ==
                                            1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                    n_config_relax, 1, 1).reshape((n_config_relax * config.hidden_size, config.hidden_size))
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight',
                           'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(
                        0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.view(
                    n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight',
                           'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.view(
                        n_state_relax, config.hidden_size).select(0, _task_idx)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, vis_feats, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, len_vis_input=49):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        # hack to load vis feats
        embedding_output = self.embeddings(vis_feats, input_ids, token_type_ids, len_vis_input=len_vis_input)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def forward(self, vis_feats,input_ids, token_type_ids, position_ids, attention_mask,
                prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True,
                len_vis_input=49):

        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
                vis_feats,input_ids, token_type_ids, 
                position_ids,len_vis_input=len_vis_input)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class BertForPreTraining(PreTrainedBertModel):

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score




""" for VD-BERT, based on UniLM """


class BertForIGLUTrain(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=18,
                 visdial_v='1.0', loss_type='mlm',float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss=''):
        super(BertForIGLUTrain, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.vis_embed = GridNet()

    def forward(self, vis_feats,input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2):

        vis_feats = self.vis_embed(vis_feats)  # image region features

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        # zero out vis_masked_pos
        if mask_image_regions:
            vis_feat_mask = vis_masked_pos.new(*vis_feats.size()[:2], 1).fill_(0).byte()
            for bb in range(vis_masked_pos.size(0)):
                for pp in range(vis_masked_pos.size(1)):
                    vis_feat_mask[bb, vis_masked_pos[bb, pp] - 1] = 1
            sequence_output, pooled_output = self.bert(vis_feats.masked_fill(vis_feat_mask, 0.),
                                                       input_ids, token_type_ids,
                                                       attention_mask, output_all_encoded_layers=False,
                                                       len_vis_input=self.len_vis_input)
        else:
            sequence_output, pooled_output = self.bert(vis_feats, input_ids, token_type_ids,
                                                       attention_mask, output_all_encoded_layers=False,
                                                       len_vis_input=self.len_vis_input)

        if masked_lm_labels is None or next_sentence_label is None:
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, task_idx=task_idx)
            return prediction_scores, seq_relationship_score

        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            # Ruotian Luo's drop worst
            keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

            # denominator = torch.sum(mask) + 1e-5
            # return (loss / denominator).sum()
            denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
            return (keep_loss / denominator).sum()

        # masked lm
        if self.loss_type == 'mlm':
            if masked_pos.numel() == 0:
                # hack to avoid empty masked_pos during training for now
                masked_lm_loss = pooled_output.new(1).fill_(0)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                masked_lm_loss = loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights, drop_worst_ratio)
            next_sentence_loss = masked_lm_loss.new(1).fill_(0)
        else:
            raise NotImplementedError

        if mask_image_regions:
            # Selfie-like pretext
            masked_vis_feats = torch.gather(vis_feats, 1,
                                            (vis_masked_pos - 1).unsqueeze(-1).expand((-1, -1, vis_feats.size(-1))))

            if self.enable_butd:
                masked_pos_enc = torch.gather(vis_pe, 1,
                                              (vis_masked_pos - 1).unsqueeze(-1).expand((-1, -1, vis_pe.size(-1))))
            else:
                masked_pos_enc = self.bert.embeddings.position_embeddings(vis_masked_pos)

            masked_pos_enc += pooled_output.unsqueeze(1).expand_as(masked_pos_enc)
            assert (masked_vis_feats.size() == masked_pos_enc.size())
            sim_mat = torch.matmul(masked_pos_enc, masked_vis_feats.permute(0, 2, 1).contiguous())
            sim_mat = F.log_softmax(sim_mat, dim=-1)
            vis_pretext_loss = []
            for i in range(sim_mat.size(0)):
                vis_pretext_loss.append(sim_mat[i].diag().mean().view(1) * -1.)  # cross entropy for ones
            vis_pretext_loss = torch.cat(vis_pretext_loss).mean()
        else:
            vis_pretext_loss = masked_lm_loss.new(1).fill_(0)

        return masked_lm_loss, vis_pretext_loss, next_sentence_loss

""" for VD-BERT, based on UniLM """

class BertForIGLUGen(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, mask_word_id=0, num_labels=2, search_beam_size=1, length_penalty=1.0, eos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0, enable_butd=False,
                 len_vis_input=36, tokenizer=None, decode_verbose=False, visdial_v='1.0'):
        super(BertForIGLUGen, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        self.tokenizer = tokenizer
        self.decode_verbose = decode_verbose
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.visdial_v = visdial_v

        # will not be initialized when loading BERT weights
        self.vis_embed = GridNet()

    def forward(self, vis_feats,input_ids, token_type_ids, position_ids, attention_mask,task_idx=None):
        vis_feats = self.vis_embed(vis_feats)  # image region features (batch_size, 100, 768)
        batch_size, vis_len, hidden_size = vis_feats.size()
        input_length = input_ids.shape[-1]
        output_length = token_type_ids.shape[-1]
        num_options = 1
        batch_size,max_tgt_length = token_type_ids.size()

        vis_feats = vis_feats.view(batch_size, 1, vis_len, hidden_size)
        vis_feats = vis_feats.view(-1, vis_len, hidden_size)

        input_ids = input_ids.view(batch_size, 1, input_length)
        input_ids = input_ids.view(batch_size * num_options, input_length)

        token_type_ids = token_type_ids.view(batch_size, 1, output_length)
        token_type_ids = token_type_ids.view(batch_size * num_options, output_length)

        position_ids = position_ids.view(batch_size, 1, output_length)
        position_ids = position_ids.view(batch_size, output_length)

        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.view(batch_size, output_length,output_length)

        
        output_ids = []
        output_probs = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id
        next_pos = input_length
        output_decode_step = 0
        
        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1] 
            
            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats,x_input_ids, curr_token_type_ids, curr_position_ids,
                          curr_attention_mask, prev_embedding=prev_embedding,
                          prev_encoded_layers=prev_encoded_layers,
                          output_all_encoded_layers=True, len_vis_input=self.len_vis_input)
            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)

            max_probs, max_ids = torch.max(prediction_scores, dim=-1)
            output_ids.append(max_ids)
            output_probs.append(max_probs)

            if prev_embedding is None:
                prev_embedding = new_embedding[:, :-1, :]

            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)

            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :]
                                       for x in new_encoded_layers]

            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]

            next_pos += 1
            output_decode_step += 1
        return torch.cat(output_ids, dim=1), torch.cat(output_probs, dim=1)



