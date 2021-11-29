import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data
from os import listdir, makedirs, remove
import os.path as osp
import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import h5py
import torch.nn as nn

import logging
from datetime import datetime
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import random
# CHECK GPU FOR PYTORCH
logging.info(f"{torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
logging.info(f"device name {device}")


from torch_geometric.utils import remove_self_loops, add_self_loops
import h5py
import torch.nn as nn

class DatasetGenerator(Dataset):
    def __init__(self,
                 source_dir=None,set_str= 'train',
                seed=None):
        self.hdf_file = h5py.File(f"/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/datagen_1295_moe_agg_queue_cdf_pdf.hdf5", "r")
        self.set_str = set_str
        
        self.inp = {'stp':['DET_LG25_S_T_0.xml', 'DET_LG25_S_T_1.xml'], 'adv':['DET_LG25_A_0.xml', 'DET_LG25_A_1.xml', 'DET_LG25_A_2.xml']
     
                     ,'sig':['sig_timing'], 
                     'queue':['queue_len']}
        self.INP_LIST = ['adv','stp', 'sig']
        self.OP_LIST = ['wait_time_pdf']
        self.Q_LIST = ['queue_len']
        self.all_nodes = [k for j in  self.INP_LIST for k in self.inp[j]]+ ['queue_len']
        self.connections =  ['DET_LG25_A_0.xml-DET_LG25_S_T_0.xml','DET_LG25_A_0.xml-DET_LG25_S_T_1.xml',
               'DET_LG25_A_1.xml-DET_LG25_S_T_0.xml', 'DET_LG25_A_1.xml-DET_LG25_S_T_1.xml',
              'DET_LG25_A_2.xml-DET_LG25_S_T_0.xml', 'DET_LG25_A_2.xml-DET_LG25_S_T_1.xml',
              ]
        
        self.edge_index = self.return_edge_index()
        self.data_size = self.len()
        
        if seed is None:
            self.rg = Generator(PCG64())
        else:
            self.rg = Generator(PCG64(seed))
        

        super(DatasetGenerator, self).__init__()
        
    
#     @property
#     def raw_file_names(self):
#         return self.source_data_files

#     @property
#     def processed_file_names(self):
#         return self._processed_file_names

    def len(self):
        test_key = list(self.hdf_file.keys())[0] 
        data_size = self.hdf_file[test_key][self.set_str].shape[0] -100
        return data_size
    
    def create_torch_data(self):
        self._processed_file_names = []
        count = 0
        for i in tqdm(range(self.str_idx,self.end_idx)):
            for k,v in self.all_dir.items():
                x,y = self.return_X_Y('train', k)
                edge_index = self.return_edge_index(x.shape[0])
                data_graph = Data(x=torch.tensor(x, dtype=torch.float).unsqueeze(1),
                                  y=torch.tensor(y, dtype=torch.float),
                                  edge_index=edge_index)

                torch.save(data_graph, self.root + '/' + TIME_SLICE_NAME + '_{}.pt'.format(count))
                self._processed_file_names.append('{}_{}.pt'.format(TIME_SLICE_NAME, count))
                count+=1

    def get(self, idx):
#         logging.info(f"get data called")
        x,y = self.return_X_Y(self.set_str, idx)
        data_graph = Data(x=torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                          y=torch.tensor(y, dtype=torch.float),
                          edge_index=self.edge_index)

#         data = torch.load(self.root +'/' + TIME_SLICE_NAME + '_{}.pt'.format(idx))\
#         logging.info(f"***** data loaded *****")
        return data_graph
    
 

                
    def return_edge_index(self):
        d = pd.DataFrame(0, index= self.all_nodes, columns= self.all_nodes)
        
        for n1 in self.all_nodes:
            for n2 in self.all_nodes:
                join_str = n1+'-'+n2
                join_str_rev = n2+'-'+n1
            #         print(f"join string {join_str}")
                if(join_str in self.connections  or 'sig_timing' in join_str or join_str_rev in self.connections  or 'sig_timing' in join_str or 'queue' in join_str):
                    d[n1][n2] = 1
                     
        source_nodes = []
        target_nodes = []
        no_of_nodes = len(self.all_nodes)
        for i,n1 in enumerate(self.all_nodes):
            for j,n2 in enumerate(self.all_nodes):
                if(d[n1][n2] == 1 and i!=j):
                    source_nodes.append(i)
                    target_nodes.append(j)
        edge_index = torch.tensor([source_nodes,target_nodes], dtype=torch.long)
        return edge_index
        
    
    def return_X_Y(self,set_str,idx):
#         pos = random.randint(0, int(self.data_size)-1) 
        pos = idx
        
        tdata_inp = {k : self.hdf_file.get(f'{k}/{set_str}')[pos] for k in self.all_nodes}
        X_train = np.dstack( tuple( [ tdata_inp[k]  for k in self.all_nodes] ) )
        
        tdata_op = {k : self.hdf_file.get(f'{k}/{set_str}')[pos] for k in self.OP_LIST}
        y_train = np.hstack( tuple( [ tdata_op[k] for k in self.OP_LIST] ) ).reshape(1,-1)
#         print(f"printing shapes {X_train.shape} {y_train.shape}")
        nan_ixs = (~np.isnan(y_train).any(axis=1))
        y_train = y_train[nan_ixs]
        X_train = X_train[nan_ixs]
        y_train = y_train*10
        y_train = np.hstack((np.zeros(y_train.shape[0]).reshape(-1,1), y_train))
        return X_train.reshape(X_train.shape[2],-1), y_train
    
    

if __name__ == "__main__":
    dataset_f = DatasetGenerator(set_str= 'train')
    logging.info(f"******** data loaded ****** {dataset_f.len()}" )
    dataloader_kwargs = {'batch_size' : 1000, 'shuffle' : True }
    data_loader = DataLoader(dataset_f,**dataloader_kwargs)