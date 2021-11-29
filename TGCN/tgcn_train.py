import sys
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
from layers import STBlock,RNNEncoder, output_layer
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
# from dataloader_moe import DatasetGenerator
from dataloader_pth_files import MOEPDFDataset
from torch.utils.tensorboard import SummaryWriter
import nvidia_dlprof_pytorch_nvtx as nvtx
import argparse

from datetime import datetime
TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_moe_pdf_one_dir'

dataset =  MOEPDFDataset(root_dir=root_dir, start_index =0, end_index = 260000,set_str= 'train')

logging.info(f"******** data loaded ****** {dataset.len()}" )

dataloader_kwargs = {'batch_size' : 1500, 'shuffle' : True}
data_loader = DataLoader(dataset,**dataloader_kwargs)
# writer = SummaryWriter('runs/tgcn_experiment_2')






class TGCN():
    def __init__(self,n_input_channels,n_units_gc,n_lstm,sequence_len,out_size):
        super().__init__()
        self.n_units_gc =  72
        self.n_input_channels = 1
        self.n_heads = 7
        self.n_lstm = n_lstm
        self.sequence_len = 72
        
        self.STblock1 = STBlock(in_channels=n_input_channels,
                               n_units_gc=self.n_units_gc,
                               n_units_lstm = n_lstm,
                               sequence_len =self.sequence_len ).to(device)
        self.out_layer = output_layer(in_size = n_lstm*self.sequence_len, out_size =out_size ).to(device)
        
        
        
        
    def train(self):
        self.STblock1.train()
        self.out_layer.train()
        
    def eval(self):
        self.STblock1.eval()
        self.out_layer.eval()
        
    def state_dict(self):
        return {
            'STblock1': self.STblock1.state_dict(),
            'out_layer': self.out_layer.state_dict()
            
        }
    
    def load_state_dict(self, state_dict):
        self.STblock1.load_state_dict(state_dict['STblock1'])
        self.out_layer.load_state_dict(state_dict['out_layer'])
        
    def __call__(self, data):
        st1_out,_ = self.STblock1(data)
        st1_out = torch.cat([ gap(st1_out, data.batch)], dim=1)
        final_out = self.out_layer(st1_out.reshape(st1_out.shape[0],-1))
        return final_out

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlprof', type=bool, default=False,
                        help='enable profiling with dlprof')
    
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs to train for')
    return parser.parse_args()
        
criterion = torch.nn.MSELoss()
t_gcn =TGCN(n_input_channels =1,
            n_units_gc = 72, 
            n_lstm=100,sequence_len=72,out_size = 200)
stgcn_layer1 = t_gcn.STblock1.to(device)
stgcn_layer1_optimizer = torch.optim.Adam(stgcn_layer1.parameters(), lr=0.001)

out_layer = t_gcn.out_layer.to(device)
out_layer_optimizer = torch.optim.Adam(out_layer.parameters(), lr=0.001)

# t_gcn.load_state_dict(torch.load('pthfiles/tgcn_pdf_mse_batch_120_epoch_16.pth'))


def train(epoch):
        
        t_gcn.train()
#         with torch.autograd.profiler.emit_nvtx():
#             for epoch in range(0,epochs):
        print ('Epoch {} at {}'.format(epoch, datetime.now().isoformat()))
        for k_batch, local_batch in enumerate(data_loader):
            stgcn_layer1_optimizer.zero_grad()
            out_layer_optimizer.zero_grad()
            local_batch = local_batch.to(device)
            if( torch.isnan(local_batch.y).any()):
                logging.info(f" ****  nan found in y ****")
            out = t_gcn(local_batch).squeeze(1)
            out = torch.softmax(out,dim =1)
            loss = criterion(out, local_batch.y)
            loss.backward()
            logging.info('...with loss {} at {}'.format(loss.item(), datetime.now().isoformat()))
#             writer.add_scalar("Loss", loss.item(), epoch * len(data_loader) + k_batch)
            stgcn_layer1_optimizer.step()
            out_layer_optimizer.step()
            if(k_batch%15 == 0):
                DICT_SAVE = t_gcn.state_dict()


def main():
    args = parse_args()
    
    # DLProf - Init PyProf
    if args.dlprof:
        nvtx.init(enable_function_stack=True)
        logging.info(f"starting training with dlprof enabled and epochs 1" )
        # Set num epochs to 1 if DLProf is enabled
        args.epochs = 1
    for epoch in range(args.epochs):    
        if args.dlprof:
            with torch.autograd.profiler.emit_nvtx():
                train(epoch)
        else:
            logging.info(f"starting training with  no profiling " )
            train(epoch)
        
if __name__ == '__main__':
    main()        
    
#             torch.save(t_gcn.state_dict(), f'pthfiles/tgcn_pdf_mse_batch_{k_batch}_epoch_{epoch}.pth')