import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TAGConv, GATConv
from os import listdir, makedirs, remove
import os.path as osp
from torch_geometric.nn import TopKPooling
import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Data
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
IDX_CHANNEL = 1
IDX_SPATIAL = 0
IDX_TEMPORAL = 2


class GCNLayer(torch.nn.Module):
    def __init__(self,in_channels, out_channels):
        super(GCNLayer, self).__init__()

        self.conv1 = GCNConv(in_channels, out_channels)
        
        self.conv2 = GCNConv(out_channels, out_channels)
        
        self.conv3 = GCNConv(out_channels, out_channels)
        
        self.pool1 = TopKPooling(20, ratio=0.8)
        self.pool2 = TopKPooling(20, ratio=0.8)
        self.pool3 = TopKPooling(20, ratio=0.8)
        
  
    def forward(self, data):
        total = []
        for k_t in range(data.x.shape[IDX_TEMPORAL]):
            x_s, edge_index, batch = data.x, data.edge_index, data.batch
            x = x_s[:,:,k_t]
            x = F.relu(self.conv1(x, edge_index))
#             x, edge_index, _, batch, _,_ = self.pool1(x, edge_index, None, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = F.relu(self.conv2(x, edge_index))
           
#             x, edge_index, _, batch, _,_ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
                  
            x = F.relu(self.conv3(x, edge_index))
#             x, edge_index, _, batch, _,_ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
   
            x = x1 + x2 + x3

            total.append(x)
        nodes_times = torch.stack(total, dim=IDX_TEMPORAL)
        

        return nodes_times
        
        
        
class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers = rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        
    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0) , self.hidden_size, device=device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        return gru_out, hidden.squeeze(0)
    
    
    
class AttentionDecoderCell(nn.Module):
    def __init__(self, hidden_size, sequence_len):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + 1, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        
    def forward(self, encoder_output, prev_hidden, y):
#         attention_input = torch.cat((prev_hidden, y), axis=1)
#         attention_weights = F.softmax(self.attention_linear(attention_input), dim=1).unsqueeze(1)
#         attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
#         print(f"attention combine shape {attention_combine.shape }")
#         rnn_hidden = self.decoder_rnn_cell(attention_combine, prev_hidden)
        rnn_hidden = self.decoder_rnn_cell(encoder_output[:,-1,:], prev_hidden)
        output = self.out(rnn_hidden)
        return output, rnn_hidden
    
class output_layer(nn.Module):
    def __init__(self, in_size,out_size):
        super(output_layer, self).__init__()
        self.linear1 = nn.Linear(in_size,64)
        self.linear2 = nn.Linear(64,out_size)

    def forward(self, x):
        x_l1 = F.relu(self.linear1(x))
        x_l2 = F.relu(self.linear2(x_l1))
        return x_l2