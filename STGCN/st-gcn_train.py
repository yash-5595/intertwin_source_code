import logging
import torch
from datetime import datetime
logging.basicConfig(level=logging.INFO)
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data
logging.info(f"{torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



from layers import STGCN, output_layer
# from dataloader_moe import DatasetGenerator
from dataloader_pth_files import MOEPDFDataset

from datetime import datetime
TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_moe_pdf_one_dir'



dataset =  MOEPDFDataset(root_dir=root_dir, start_index =10000, end_index = 260000,set_str= 'train')

logging.info(f"******** data loaded ****** {dataset.len()}" )

dataloader_kwargs = {'batch_size' : 2500, 'shuffle' : True}
data_loader = DataLoader(dataset,**dataloader_kwargs)

n_temporal =  72
n_input_channels = 1
out_size = 200


class STGCNWrapper():
    def __init__(self, n_temporal, n_input_channels, out_size):
        super().__init__()
        self.n_temporal =  96
        self.n_input_channels = 1
        self.co_spatial = 16
        self.co_temporal = 64
        self.time_conv_length = 3
        self.out_size = 200
        
        self.gcn_layer = STGCN(n_temporal, n_input_channels,
                 self.co_temporal, self.co_spatial, self.time_conv_length).to(device)
        self.output_layer = output_layer( n_temporal-8,64,self.co_temporal,self.time_conv_length,self.out_size ).to(device)
        
        
    def train(self):
        self.gcn_layer.train()
        self.output_layer.train()
        
    def eval(self):
        self.gcn_layer.eval()
        self.output_layer.eval()
        
    def state_dict(self):
        return {
            'gcn_layer': self.gcn_layer.state_dict(),
            'output_layer': self.output_layer.state_dict()
            
        }
    
    def load_state_dict(self, state_dict):
        self.gcn_layer.load_state_dict(state_dict['gcn_layer'])
        self.output_layer.load_state_dict(state_dict['output_layer'])
        
    def __call__(self, data):
        gc_out = self.gcn_layer(data)
        out = self.output_layer(gc_out)
        return out
        
        
        
criterion = torch.nn.MSELoss()
stgcn_module =STGCNWrapper(n_temporal, n_input_channels, out_size)
stgcn_layer = stgcn_module.gcn_layer.to(device)
gc_optimizer = torch.optim.Adam(stgcn_layer.parameters(), lr=0.001)

out_layer = stgcn_module.output_layer.to(device)
out_optimizer = torch.optim.Adam(out_layer.parameters(), lr=0.001)

stgcn_module.load_state_dict(torch.load('pthfiles/stgcn_pdf_mse_softmax_batch_90_epoch_12.pth'))

for epoch in range(13,100):
    print ('Epoch {} at {}'.format(epoch, datetime.now().isoformat()))
    for k_batch, local_batch in enumerate(data_loader):
        gc_optimizer.zero_grad()
        out_optimizer.zero_grad()
        local_batch = local_batch.to(device)
        if( torch.isnan(local_batch.y).any()):
            logging.info(f" ****  nan found in y ****")
        out = stgcn_module(local_batch).squeeze(1)
        out = torch.softmax(out,dim =1)
        loss = criterion(out, local_batch.y)
        loss.backward()
        logging.info('...with loss {} at {}'.format(loss.item(), datetime.now().isoformat()))
        gc_optimizer.step()
        out_optimizer.step()
        if(k_batch%15 == 0):
            DICT_SAVE = stgcn_module.state_dict()
            torch.save(stgcn_module.state_dict(), f'pthfiles/stgcn_pdf_mse_softmax_batch_{k_batch}_epoch_{epoch}.pth')
