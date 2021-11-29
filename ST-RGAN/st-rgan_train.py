from dataloader_pth_files import MOEPDFDataset
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

from datetime import datetime
TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_moe_pdf_one_dir'

dataset =  MOEPDFDataset(root_dir=root_dir, start_index =0, end_index = 260000,set_str= 'train')

logging.info(f"******** data loaded ****** {dataset.len()}" )

dataloader_kwargs = {'batch_size' : 100, 'shuffle' : True}
data_loader = DataLoader(dataset,**dataloader_kwargs)

class STRGAN():
    def __init__(self,n_input_channels,n_units_gc, n_heads,n_lstm,sequence_len,out_size):
        super().__init__()
        self.n_units_gc =  72
        self.n_input_channels = 1
        self.n_heads = 7
        self.n_lstm = n_lstm
        self.sequence_len = 72
        
        self.STblock1 = STBlock(in_channels=n_input_channels,
                               n_units_gc=self.n_units_gc,
                               n_heads= self.n_heads,
                               n_units_lstm = n_lstm,
                               sequence_len =self.sequence_len ).to(device)
        self.STblock2 = STBlock(in_channels=self.n_lstm,
                               n_units_gc=self.n_units_gc,
                               n_heads= self.n_heads,
                               n_units_lstm = 128,
                               sequence_len =self.sequence_len ).to(device)
        self.out_layer = output_layer(in_size = 128*self.sequence_len, out_size =out_size ).to(device)
        
        
        
        
    def train(self):
        self.STblock1.train()
        self.STblock2.train()
        self.out_layer.train()
        
    def eval(self):
        self.STblock1.eval()
        self.STblock2.eval()
        self.out_layer.eval()
        
    def state_dict(self):
        return {
            'STblock1': self.STblock1.state_dict(),
            'STblock2': self.STblock2.state_dict(),
            'out_layer': self.out_layer.state_dict()
            
        }
    
    def load_state_dict(self, state_dict):
        self.STblock1.load_state_dict(state_dict['STblock1'])
        self.STblock2.load_state_dict(state_dict['STblock2'])
        self.out_layer.load_state_dict(state_dict['out_layer'])
        
    def __call__(self, data):
        st1_out,_ = self.STblock1(data)
        st1_out = st1_out.transpose(1, 2)
        st2_in = Data(st1_out,data.edge_index)
        st2_out,_ = self.STblock2(st2_in)
        st2_out = torch.cat([ gap(st2_out, data.batch)], dim=1)
        final_out = self.out_layer(st2_out.reshape(st2_out.shape[0],-1))
        return final_out
criterion = torch.nn.MSELoss()

st_rgan =STRGAN(n_input_channels =1 ,n_units_gc = 72, n_heads=7,n_lstm=100,sequence_len=72,out_size = 200)

stgcn_layer1 = st_rgan.STblock1.to(device)
stgcn_layer1_optimizer = torch.optim.Adam(stgcn_layer1.parameters(), lr=0.001)


stgcn_layer2 = st_rgan.STblock2.to(device)
stgcn_layer2_optimizer = torch.optim.Adam(stgcn_layer2.parameters(), lr=0.001)

out_layer = st_rgan.out_layer.to(device)
out_layer_optimizer = torch.optim.Adam(out_layer.parameters(), lr=0.001)


st_rgan.train()
for epoch in range(1,100):
    print ('Epoch {} at {}'.format(epoch, datetime.now().isoformat()))
    for k_batch, local_batch in enumerate(data_loader):
        stgcn_layer1_optimizer.zero_grad()
        stgcn_layer2_optimizer.zero_grad()
        out_layer_optimizer.zero_grad()
        local_batch = local_batch.to(device)
        if( torch.isnan(local_batch.y).any()):
            logging.info(f" ****  nan found in y ****")
        out = st_rgan(local_batch).squeeze(1)
        out = torch.softmax(out,dim =1)
        loss = criterion(out, local_batch.y)
        loss.backward()
        logging.info('...with loss {} at {}'.format(loss.item(), datetime.now().isoformat()))
        stgcn_layer1_optimizer.step()
        stgcn_layer2_optimizer.step()
        out_layer_optimizer.step()
        if(k_batch%50 == 0):
            DICT_SAVE = st_rgan.state_dict()
            torch.save(st_rgan.state_dict(), f'pthfiles/st_rgan_pdf_mse_batch_softmax{k_batch}_epoch_{epoch}.pth')
