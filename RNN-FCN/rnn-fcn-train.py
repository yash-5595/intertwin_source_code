import logging
import torch
import torch.nn as nn
from datetime import datetime
logging.basicConfig(level=logging.INFO)

logging.info(f"{torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data


TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_moe_pdf_one_dir'
from dataloader_pth_files import MOEPDFDataset
import torch.nn.functional as F
dataset =  MOEPDFDataset(root_dir=root_dir, start_index =0, end_index = 260000,set_str= 'train')

logging.info(f"******** data loaded ****** {dataset.len()}" )

dataloader_kwargs = {'batch_size' : 5000, 'shuffle' : True}
data_loader = DataLoader(dataset,**dataloader_kwargs)


class GRU_Net(nn.Module):
    def __init__(self, n_features,seq_length ):
        super().__init__() # Init super class too
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 100# number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
        
        self.gru1 = nn.GRU(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True).to(device)
        
        
       
        
        
#         dense connected layers 
        self.fc1 = nn.Linear(self.n_hidden, 128).to(device)
#         self.bn1 = nn.BatchNorm1d(64).to(device)
        self.dr1 = nn.Dropout(0.25).to(device)
        
        self.fc2 = nn.Linear(128, 64).to(device)
#         self.bn2 = nn.BatchNorm1d(56).to(device)
        self.dr2 = nn.Dropout(0.25).to(device)
        
        self.fc3 = nn.Linear(64, 200).to(device)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        self.hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device)


    def forward(self, data):
        
        x = data.x.reshape(-1,72,7)
#        
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.gru1(x,self.hidden)
        
        x1 = lstm_out.contiguous().view(batch_size,-1)
        x2 = self.hidden.contiguous().view(batch_size,-1)
#         return x1, x2
        
#         x = torch.flatten(x,1)
    
        x2 = F.relu(self.dr1((self.fc1(x2))))
        x2 = F.relu(self.dr2((self.fc2(x2))))    
        x2 =(self.fc3(x2))
    
        return x2

agg_factor = 1
net = GRU_Net(n_features = 7,seq_length=200)    
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
pytorch_total_params
criterion = torch.nn.MSELoss()

net.train()
for epoch in range(1,100):
    print ('Epoch {} at {}'.format(epoch, datetime.now().isoformat()))
    for k_batch, local_batch in enumerate(data_loader):
        optimizer.zero_grad()
        net.init_hidden(int(local_batch.x.shape[0]/7))
        local_batch = local_batch.to(device)
        if( torch.isnan(local_batch.y).any()):
            logging.info(f" ****  nan found in y ****")
        out = net(local_batch).squeeze(1)
        out = torch.softmax(out,dim =1)
        loss = criterion(out, local_batch.y)
        loss.backward()
        logging.info('...with loss {} at {}'.format(loss.item(), datetime.now().isoformat()))
        optimizer.step()
        if(k_batch%15 == 0):
            torch.save(net, f'pthfiles/moe_pdf_mse_rnnfcn_softmax_{k_batch}_epoch_{epoch}.pth.pth')