from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
from collections import defaultdict
import random
import coloredlogs, logging
from colorama import Fore,Style
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer, BertPreTrainedModel)

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_bert  import BertEmbeddings

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from torch.nn import CrossEntropyLoss, MSELoss



from utils_classify import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',level='INFO',datefmt='%m/%d %H:%M:%S',logger=logger)




#MLPForSequenceClassificationj$
class BiLSTMForSequenceClassification(nn.Module):
    def __init__(self, config, weights=None):
        super(BiLSTMForSequenceClassification, self).__init__()

        # from pdb import set_trace; set_trace()
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        # self.bert = BertModel(config)

        # self.embeddings = BertEmbeddings(config)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) #, padding_idx=0)
        # self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.encoder = nn.LSTM(config.embedding_size, config.hidden_size)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.init_weights()



    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, batch_size=None):

        # from pdb import set_trace; set_trace()
        embeddings = self.word_embeddings(input_ids)
        embeddings  = embeddings.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
                h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
                c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
        else:
                h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
                c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.encoder(embeddings, (h_0, c_0))
        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)


        outputs = (logits,) #+ outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        # print(loss.item())
        return outputs  # (loss), logits, (hidden_states), (attentions)


