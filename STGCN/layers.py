import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TAGConv
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

IDX_CHANNEL = 1
IDX_SPATIAL = 0
IDX_TEMPORAL = 2


class SpatioTemporal(torch.nn.Module):
    '''Base Class for the Spatio Temporal classes
    '''
    def __init__(self, n_temporal,
                 channel_inputs, channel_outputs):

        super(SpatioTemporal, self).__init__()

        self.n_temporal = n_temporal

        self.channel_inputs = channel_inputs
        self.channel_outputs = channel_outputs
        
        
class SpatialGraphConv(SpatioTemporal):
    '''Convolution in spatial dimension with graph convolution
        The graph convolution is applied to all temporal layers, so one set of parameters apply to all
        temporal layers. The graph convolution is based on the graph Laplacian, and hence only spatial units
        that are connected can contribute to each others outputs after convolution.
        The input dimension is (N, T, Cin) and the output dimension is (N, T, Cout), where N is the number of
        spatial units, T is the number of temporal units per spatial unit, Cin the number of
        channels of the input, and Cout the number of channels of the output.
        Args:
            n_spatial (int): Number of spatial units in mode, equal to number of nodes in graph
            n_temporal (int): Number of contiguous temporal units per spatial unit
            channel_inputs (int): Number of channels of input per node
            channel_outputs (int): Number of channels of output per node
            graph_conv_type (str): Type of graph convolution to use. Supported types are 'GCNConv' and 'TAGConv'
            graph_conv_kwargs (dict): Keyword arguments in addition to the input and output channel data to the
                graph convolution class
    '''
    def __init__(self, n_temporal, channel_inputs, channel_outputs,
                 graph_conv_type, graph_conv_kwargs):

        super(SpatialGraphConv, self).__init__(n_temporal, channel_inputs, channel_outputs)

        if graph_conv_type == 'GCNConv':
            self.gcn = GCNConv(self.channel_inputs, self.channel_outputs)

        elif graph_conv_type == 'TAGConv':
            self.gcn = TAGConv(self.channel_inputs, self.channel_outputs, **graph_conv_kwargs)

        else:
            raise ValueError('Unknown graph convolution type encountered: {}'.format(graph_conv_type))

    def forward(self, data):

        assert self.channel_inputs == data.x.shape[IDX_CHANNEL]

        x_s = data.x
        total = []
        for k_t in range(data.x.shape[IDX_TEMPORAL]):
            x_t = x_s[:,:,k_t]
            y_t = self.gcn(x_t, data.edge_index, data.edge_attr)
            y_t = F.relu(y_t)
            total.append(y_t)

        nodes_times = torch.stack(total, dim=IDX_TEMPORAL)

        return nodes_times
    
    
class Time1dConvGLU(SpatioTemporal):
    '''Convolution along time dimension with gated linear unit as output activation
    The temporal convolution applies a 1D convolution of a set kernel size. The temporal convolution is applied
    to all spatial units in the forward operation. That is, the parameters are the same for all spatial
    units. The output of the 1D convolution is passed through a gated linear unit, hence the channel output is
    cut in half.
    The input dimension is (N, T, Cin) and the output dimension is (N, T-k+1, Cout), where N is the number of
    spatial units, T is the number of temporal units per spatial unit, k is the kernel size, Cin the number of
    channels of the input, and Cout the number of channels of the output.
    Number of parameters scale as: 2 * Cout + 2 * Cout * Cin * k
    Args:
        n_spatial (int): Number of spatial units in mode, equal to number of nodes in graph
        n_temporal (int): Number of contiguous temporal units per spatial unit
        channel_inputs (int): Number of channels of input per node
        channel_outputs (int): Number of channels of output per node
        time_convolution_length (int): The kernel size of the time convolution window
    '''
    def __init__(self, n_temporal, channel_inputs, channel_outputs,
                 time_convolution_length):

        super(Time1dConvGLU, self).__init__(n_temporal, channel_inputs, channel_outputs)

        self.one_d_conv_1 = torch.nn.Conv1d(in_channels=self.channel_inputs,
                                            out_channels=2*self.channel_outputs,
                                            kernel_size=time_convolution_length,
                                            groups=1)

        n_temporal_out = self.n_temporal - time_convolution_length + 1
        assert n_temporal_out > 0

    def forward(self, x):
        '''Forward operation. Apply the same temporal convolution to all spatial units'''

        total = []
        for k_node, tensor_node in enumerate(x.split(1, dim=IDX_SPATIAL)):
            pq_out = self.one_d_conv_1(tensor_node)
            y_conv_and_gated_out = F.glu(pq_out, dim=1)
            total.append(y_conv_and_gated_out)

        nodes_spatial = torch.cat(total, dim=IDX_SPATIAL)

        return nodes_spatial
    
    
class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)
    
    
    
class STGCN(torch.nn.Module):
    def __init__(self, n_temporal_dim, n_input_channels,
                 co_temporal=64, co_spatial=16, time_conv_length=3,
                 graph_conv_type='GCNConv', graph_conv_kwargs={}):
        super(STGCN, self).__init__()
        self.n_temporal = n_temporal_dim
        self.n_input_channels = n_input_channels
        self.co_temporal = co_temporal
        self.co_spatial = co_spatial
        
        
        self.model_t_1a = Time1dConvGLU( n_temporal_dim,
                                        channel_inputs=n_input_channels,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.model_s_1 = SpatialGraphConv( n_temporal_dim - time_conv_length + 1,
                                          channel_inputs=co_temporal,
                                          channel_outputs=co_spatial,
                                          graph_conv_type=graph_conv_type,
                                          graph_conv_kwargs=graph_conv_kwargs)
        
        self.model_t_1b = Time1dConvGLU( n_temporal_dim - time_conv_length + 1,
                                        channel_inputs=co_spatial,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        
        self.model_t_2a = Time1dConvGLU( n_temporal_dim - 2 * time_conv_length + 2,
                                        channel_inputs=co_temporal,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.model_s_2 = SpatialGraphConv( n_temporal_dim - 3 * time_conv_length + 3,
                                          channel_inputs=co_temporal,
                                          channel_outputs=co_spatial,
                                          graph_conv_type=graph_conv_type,
                                          graph_conv_kwargs=graph_conv_kwargs)
        self.model_t_2b = Time1dConvGLU( n_temporal_dim - 3 * time_conv_length + 3,
                                        channel_inputs=co_spatial,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        
    def forward(self, data_graph):
        data_step_0 = data_graph
        edge_index = data_graph.edge_index
        batch = data_graph.batch
        
        
        data_step_0_x = data_step_0.x
#         print(f"data_step_0_x {data_step_0_x.shape}")
        data_step_1_x = self.model_t_1a(data_step_0_x)
#         print(f"data_step_1_x {data_step_1_x.shape}")
        data_step_1 = Data(data_step_1_x, edge_index)
        data_step_2_x = self.model_s_1(data_step_1)
#         print(f"data_step_2_x {data_step_2_x.shape}")
        x1 = torch.cat([ gap(data_step_2_x, batch)], dim=1)
#         print(f" x1 {x1.shape}")
        data_step_3_x = self.model_t_1b(data_step_2_x)
#         print(f"end of first STConv block data_step_3_x {data_step_3_x.shape}")
        
        
        
        data_step_4_x = self.model_t_2a(data_step_3_x)
        data_step_4 = Data(data_step_4_x, edge_index)
        data_step_5_x = self.model_s_2(data_step_4)
        x2 = torch.cat([ gap(data_step_5_x, batch)], dim=1)
        data_step_3_x = self.model_t_2b(x2)
#         print(f"end of second STConv blocl x2 {data_step_3_x.shape}")
        return data_step_3_x
    
class output_layer(nn.Module):
    def __init__(self, n_temporal_dim, n_input_channels,co_temporal,time_conv_length,out_size):
        super(output_layer, self).__init__()
        self.tconv1 = Time1dConvGLU( n_temporal_dim,
                                        channel_inputs=n_input_channels,
                                        channel_outputs=co_temporal,
                                        time_convolution_length=time_conv_length)
        self.fc = fully_conv_layer(co_temporal)
        self.linear1 = nn.Linear(n_temporal_dim - time_conv_length + 1,64)
        self.linear2 = nn.Linear(64,out_size)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_t2 = self.fc(x_t1.unsqueeze(-1)).squeeze(-1)
        x_l1 = F.relu(self.linear1(x_t2))
        x_l2 = F.relu(self.linear2(x_l1))
        return x_l2
        