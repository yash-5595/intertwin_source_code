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
from layers import GCNLayer,RNNEncoder, AttentionDecoderCell, output_layer
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp


import torch.nn.functional as F


TIME_SLICE_NAME = 'exemplarid'
root_dir = '/blue/ranka/yashaswikarnati/yash_simulation_data/datagen_multilane/train_data_moe_pdf_one_dir'

from dataloader_pth_files import MOEPDFDataset

dataset =  MOEPDFDataset(root_dir=root_dir, start_index =0, end_index = 260000,set_str= 'train')

logging.info(f"******** data loaded ****** {dataset.len()}" )

dataloader_kwargs = {'batch_size' : 1500, 'shuffle' : True, 'num_workers':5}
data_loader = DataLoader(dataset,**dataloader_kwargs)

IDX_CHANNEL = 1
IDX_SPATIAL = 0
IDX_TEMPORAL = 2

class EncoderDecoderWrapper():
    def __init__(self, in_channels, out_channels, sequence_len,rnn_hid_size, output_size=3, teacher_forcing=0.0):
        super().__init__()
        
        self.gcn_layer = GCNLayer(in_channels, out_channels).to(device)
        self.encoder = RNNEncoder(rnn_num_layers=1, input_feature_len=7,
                                  sequence_len=sequence_len, hidden_size=rnn_hid_size).to(device)
                                  
        self.decoder_cell = AttentionDecoderCell( hidden_size = rnn_hid_size, 
                                                 sequence_len = sequence_len).to(device)
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_len = sequence_len
        
    def train(self):
        self.gcn_layer.train()
        self.encoder.train()
        self.decoder_cell.train()
        
    def eval(self):
        self.gcn_layer.eval()
        self.encoder.eval()
        self.decoder_cell.eval()
        
    def state_dict(self):
        return {
            'gcn_layer': self.gcn_layer.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder_cell': self.decoder_cell.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.gcn_layer.load_state_dict(state_dict['gcn_layer'])
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder_cell.load_state_dict(state_dict['decoder_cell'])

    def __call__(self, data):
#         gc_out = self.gcn_layer(data)
#         concat_x = torch.cat([data.x.reshape(-1,7,self.sequence_len),gc_out], dim=IDX_CHANNEL)
        yb = data.y
#         input_seq = concat_x.transpose(1, 2)
#         logging.info(f"input seq. {input_seq.shape}")
        input_seq = data.x.reshape(-1,72,7)
        encoder_output, encoder_hidden = self.encoder(input_seq)
#         logging.info(f"encoder output {encoder_output.shape} hidden {encoder_hidden.shape}")
        
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(input_seq.size(0), self.output_size, device=device)
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)
        y_prev = yb[:,0].unsqueeze(1)
        
        outputs[:, 0] = yb[:,0]
        for i in range(1,self.output_size):
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = yb[:, i-1].unsqueeze(1)
#             logging.info(f"y_prev  {y_prev.shape}")
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
#             logging.info(f"rnn output {rnn_output.shape}")
#             logging.info(f"prev hidden {prev_hidden.shape}")
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        return F.softmax(outputs, dim =1)
    

gc_edtam = EncoderDecoderWrapper(in_channels = 1, out_channels = 20, sequence_len = 72,rnn_hid_size = 50, output_size=200, teacher_forcing=0.0)


gcn_layer = gc_edtam.gcn_layer.to(device)
gc_optimizer = torch.optim.Adam(gcn_layer.parameters(), lr=0.001)

encoder = gc_edtam.encoder.to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder = gc_edtam.decoder_cell.to(device)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

criterion = torch.nn.MSELoss()
gc_edtam.train()
for epoch in range(1,500):
    print ('Epoch {} at {}'.format(epoch, datetime.now().isoformat()))
    for k_batch, local_batch in enumerate(data_loader):
        gc_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        local_batch = local_batch.to(device)
        if( torch.isnan(local_batch.y).any()):
            logging.info(f" ****  nan found in y ****")
        out = gc_edtam(local_batch).squeeze(1)
#         out = torch.softmax(out,dim =1)
        loss = criterion(out, local_batch.y)
        loss.backward()
        logging.info('...with loss {} at {}'.format(loss.item(), datetime.now().isoformat()))
        gc_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()
        if(k_batch%15 == 0):
            DICT_SAVE = gc_edtam.state_dict()
            torch.save(gc_edtam.state_dict(), f'pthfiles/gc-edtam_no_gcn_pdf_mse_batch_{k_batch}_epoch_{epoch}.pth')