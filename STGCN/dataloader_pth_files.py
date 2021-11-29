import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data

from os import listdir, makedirs, remove
import os.path as osp
import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64
import h5py
import random
from datetime import datetime
import uuid
from functools import partial
from multiprocessing.pool import ThreadPool as Pool


TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_moe_pdf_one_dir'



class MOEPDFDataset(Dataset):
#     @staticmethod
    

    
    def __init__(self, root_dir,start_index, end_index,set_str = "train",
                 create_from_source=False,
                 source_dir=None, source_data_files=None,
                 sample_size=None, stride=None,
                seed=None):
        self.root = root_dir
        self.set_str = set_str
        self.hdf_file = h5py.File(f"/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/datagen_1295_moe_agg_queue_cdf_pdf.hdf5", "r")
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
        self.str_idx = start_index
        self.end_idx= end_index
        self.no_of_cores = 25

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

        super(MOEPDFDataset, self).__init__(root_dir)
        
    
    @property
    def raw_file_names(self):
        return self.source_data_files

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        try:
            data = torch.load(self.root +'/' + TIME_SLICE_NAME + '_{}.pt'.format(idx))
        except Exception as e:
            data = torch.load(self.root +'/' + TIME_SLICE_NAME + '_{}.pt'.format(500))
        return data
    

    
    
    def create_torch_data_parallel(self):
        curr_idx = self.str_idx
        batch_size = 100
        while(curr_idx<self.end_idx-batch_size-1):
            print(f"doing for start index {curr_idx}")
            x,y = self.return_X_Y_batch(self.set_str,curr_idx,curr_idx+batch_size)
            p = Pool(self.no_of_cores)
            partial_func = partial(self.write_single_file,list(x),list(y),curr_idx )
            p.map(partial_func,list(range(curr_idx,curr_idx+batch_size)))
            p.close()
            curr_idx+=batch_size
            
    def write_single_file(self,x,y,curr_idx,index):
        #         x,y,index = params['x'],params['y'], params['index']
        try:
#             print(f"function called for index  {index}")
            data_graph = Data(x=torch.tensor(x[index-curr_idx], dtype=torch.float32).unsqueeze(1),
                              y=torch.tensor(y[index-curr_idx].reshape(1,-1), dtype=torch.float),
                              edge_index=self.edge_index)

            torch.save(data_graph, self.root + '/' + TIME_SLICE_NAME + '_{}.pt'.format(index))
        except Exception as e:
            print(f"*** failed for *** index {index})")

    
    def create_torch_data(self):
        self._processed_file_names = []
        count = 0
        for i in tqdm(range(self.str_idx,self.end_idx)):
            x,y = self.return_X_Y(self.set_str, i)
            data_graph = Data(x=torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                          y=torch.tensor(y, dtype=torch.float),
                          edge_index=self.edge_index)

            torch.save(data_graph, self.root + '/' + TIME_SLICE_NAME + '_{}.pt'.format(count))
            self._processed_file_names.append('{}_{}.pt'.format(TIME_SLICE_NAME, count))
            count+=1

                
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
    
    
    
    def return_X_Y_batch(self,set_str,start,end):
#         pos = random.randint(0, int(self.data_size)-1) 
        
        tdata_inp = {k : self.hdf_file.get(f'{k}/{set_str}')[start:end] for k in self.all_nodes}
        X_train = np.dstack( tuple( [ tdata_inp[k]  for k in self.all_nodes] ) )
        
        tdata_op = {k : self.hdf_file.get(f'{k}/{set_str}')[start:end] for k in self.OP_LIST}
        y_train = np.hstack( tuple( [ tdata_op[k] for k in self.OP_LIST] ) )
#         print(f"printing shapes {X_train.shape} {y_train.shape}")
        nan_ixs = (~np.isnan(y_train).any(axis=1))
        y_train = y_train[nan_ixs]
        X_train = X_train[nan_ixs]
        y_train = y_train*10
        y_train = np.hstack((np.zeros(y_train.shape[0]).reshape(-1,1), y_train))
        X_train = np.transpose(X_train, (0,2,1))
        return X_train, y_train

    def return_sum_one_type(self,pos, set_str, curr_dir, det_type):
        tdata = {k: hdf_file.get(f'{k}/{set_str}')[pos]  for k in curr_dir[det_type]}
        sum_arr = np.zeros((1, 96))
        for k,v in tdata.items():
            sum_arr += v
        return sum_arr
    