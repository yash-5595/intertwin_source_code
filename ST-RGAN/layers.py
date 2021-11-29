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


class STBlock(nn.Module):
    def __init__(self,in_channels, n_units_gc,n_heads, n_units_lstm, sequence_len):
        super(STBlock,self).__init__()
        
        self.n_units_gc = n_units_gc
        self.n_heads = n_heads
        self.n_units_lstm = n_units_lstm
        self.seq_len = sequence_len
        
        self.gcn = GATConv(in_channels =in_channels, out_channels  =  n_units_gc, heads = n_heads)
        
        self.rnn = RNNEncoder(input_feature_len = (n_units_gc*n_heads)+in_channels,
                sequence_len=sequence_len,
                hidden_size=n_units_lstm)
        
        
    def forward(self, data):
        x_s = data.x
        total = []
        for k_t in range(data.x.shape[IDX_TEMPORAL]):
            x_t = x_s[:,:,k_t]
            y_t = self.gcn(x_t, data.edge_index, data.edge_attr)
            y_t = F.relu(y_t)
            total.append(y_t)

            
        nodes_times = torch.stack(total, dim=IDX_TEMPORAL)
        #   adding residual connection by concatenating input x, gat output across feature dimension
        concat_x = torch.cat([x_s,nodes_times], dim=IDX_CHANNEL)
#         nodes_times = torch.cat([ gap(concat_x, data.batch)], dim=1)
        rnn_out,hid_st  = self.rnn(concat_x.transpose(1, 2))

        return rnn_out,hid_st
        
        
        
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
    
    
class output_layer(nn.Module):
    def __init__(self, in_size,out_size):
        super(output_layer, self).__init__()
        self.linear1 = nn.Linear(in_size,64)
        self.linear2 = nn.Linear(64,out_size)

    def forward(self, x):
        x_l1 = F.relu(self.linear1(x))
        x_l2 = F.relu(self.linear2(x_l1))
        return x_l2