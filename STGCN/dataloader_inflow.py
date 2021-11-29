
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.init as init
import h5py
from os import listdir, makedirs, remove
import os.path as osp
import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)

# all constants (global)
hdf_file = h5py.File(f"/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/datagen_intersection_1295_multilane.hdf5", "r")
TIME_SLICE_NAME = 'exemplarid'
test_key = list(hdf_file.keys())[0] 
data_size = hdf_file[test_key]['train'].shape[0]
data_size_test = hdf_file[test_key]['test'].shape[0]
batch_size = 1000
dir_25 = {'adv' : ['DET_DET_LG25_A_0', 'DET_DET_LG25_A_1', 'DET_DET_LG25_A_2'], 'stp':['DET_DET_LG25_S_L_0', 'DET_DET_LG25_S_T_0', 'DET_DET_LG25_S_T_1'], 'inflow' :['DET_DET_LG25_in_500_0', 'DET_DET_LG25_in_500_1', 'DET_DET_LG25_in_500_2']}
dir_16 = {'adv' : ['DET_DET_LG16_A_0', 'DET_DET_LG16_A_1', 'DET_DET_LG16_A_2', 'DET_DET_LG16_A_3'], 'stp':['DET_DET_LG16_S_L_0', 'DET_DET_LG16_S_T_0', 'DET_DET_LG16_S_T_1'], 'inflow' :['DET_DET_LG16_in_500_0', 'DET_DET_LG16_in_500_1', 'DET_DET_LG16_in_500_2', 'DET_DET_LG16_in_500_3']}
dir_47 = {'adv' : ['DET_DET_LG47_A_0', 'DET_DET_LG47_A_1', 'DET_DET_LG47_A_2'], 'stp':['DET_DET_LG47_S_L_0', 'DET_DET_LG47_S_L_1', 'DET_DET_LG47_S_T_0', 'DET_DET_LG47_S_T_1', 'DET_DET_LG47_S_T_2'], 'inflow' :['DET_DET_LG47_in_500_0', 'DET_DET_LG47_in_500_1', 'DET_DET_LG47_in_500_2']}
dir_38 = {'adv' : ['DET_DET_LG38_A_0', 'DET_DET_LG38_A_1', 'DET_DET_LG38_A_2'], 'stp':['DET_DET_LG38_S_L_0', 'DET_DET_LG38_S_L_1', 'DET_DET_LG38_S_T_0', 'DET_DET_LG38_S_T_1', 'DET_DET_LG38_S_T_2'], 'inflow' :['DET_DET_LG38_in_500_0', 'DET_DET_LG38_in_500_1', 'DET_DET_LG38_in_500_2']}


all_dir = {'dir_25':dir_25, 'dir_16':dir_16, 'dir_47':dir_47, 'dir_38':dir_38}
TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_2/'



class InfowDataset(Dataset):
    def __init__(self, root_dir,start_index, end_index,
                 create_from_source=True,
                 source_dir=None, source_data_files=None,
                 sample_size=None, stride=None,
                seed=None):
        self.root = root_dir
        self.all_dir = {'dir_25':dir_25, 'dir_16':dir_16, 'dir_47':dir_47, 'dir_38':dir_38}
        self.str_idx = start_index
        self.end_idx= end_index

        if create_from_source:
            if any([source_dir, source_data_files]) is None:
                raise ValueError('To process the dataset, source directory, source files and source graph file all required')
        self.source_dir = source_dir
        self.sample_size = sample_size
        if seed is None:
            self.rg = Generator(PCG64())
        else:
            self.rg = Generator(PCG64(seed))
        
        if create_from_source:
            if not osp.exists(root_dir):
                makedirs(root_dir)
            else:
                for ff in listdir(root_dir):
                    remove(root_dir + '/' + ff)
            self.create_torch_data()
        else:
            self._processed_file_names = [fname for fname in listdir(root_dir) if TIME_SLICE_NAME in fname]

        super(InfowDataset, self).__init__(root_dir)
        
    
    @property
    def raw_file_names(self):
        return self.source_data_files

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.root +'/' + TIME_SLICE_NAME + '_{}.pt'.format(idx))
        return data
    
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

                
    def return_edge_index(self, no_of_nodes):
        source_nodes = []
        target_nodes = []
        for i in range(no_of_nodes):
            for j in range(no_of_nodes):
                if(i!=j):
                    source_nodes.append(i)
                    target_nodes.append(j)
        edge_index = torch.tensor([source_nodes,
                                   target_nodes], dtype=torch.long)
        return edge_index
        
    
    def return_X_Y(self,set_str, direction):
        pos = random.randint(0, int(data_size/batch_size)-1) * batch_size
        INP_LIST = ['adv','stp']
        OP_LIST = ['inflow']
        curr_dir = all_dir[direction]
        inp_list = []
        for each_inp in INP_LIST:
            tdata = {k: hdf_file.get(f'{k}/{set_str}')[pos]  for k in curr_dir[each_inp]}
            x = np.vstack( tuple( [ tdata[k] for k in curr_dir[each_inp]] ) )
            inp_list.append(x)
        X_train = np.vstack(inp_list)


        tdata_op = {k :self.return_sum_one_type(pos,set_str,curr_dir, k) for k in OP_LIST}
        y_train = np.vstack( tuple( [ tdata_op[k] for k in OP_LIST] ) )

        return X_train, y_train

    def return_sum_one_type(self,pos, set_str, curr_dir, det_type):
        tdata = {k: hdf_file.get(f'{k}/{set_str}')[pos]  for k in curr_dir[det_type]}
        sum_arr = np.zeros((1, 96))
        for k,v in tdata.items():
            sum_arr += v
        return sum_arr
    


