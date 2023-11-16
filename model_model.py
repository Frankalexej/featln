import torch.nn as nn
from model_config import *

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(n_chans, n_chans)
        self.lin2 = nn.Linear(n_chans, n_chans)
        self.batch_norm = nn.BatchNorm1d(num_features=n_chans)  # <5>
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lin1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out + x

class LinPack(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinPack, self).__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(num_features=n_out)
        # self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x):
        x = self.lin(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        # x = self.dropout(x)
        return x


class ResAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, inter_dim1=INTER_DIM_1, inter_dim2=INTER_DIM_2, inter_dim3=INTER_DIM_3, latent_dim=LATENT_DIM, output_dim=OUTPUT_DIM):
        super(ResAE, self).__init__()

        self.encoder = nn.Sequential(
            LinPack(input_dim, inter_dim1), 
            ResBlock(inter_dim1), 
            # ResBlock(inter_dim1), 
            nn.Linear(inter_dim1, latent_dim), 
            nn.Sigmoid()
        )

        self.decoder =  nn.Sequential(
            LinPack(latent_dim, inter_dim1), 
            ResBlock(inter_dim1), 
            # ResBlock(inter_dim1), 
            nn.Linear(inter_dim1, output_dim),
            # nn.Sigmoid(),
        )

        # initialize the weights
        self.encoder.apply(self.init_weights)
        self.encoder.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        org_size = x.size()
        y_size = (org_size[0], org_size[1], org_size[2] // 3)
        batch = org_size[0]
        x = x.view(batch, -1)

        h = self.encoder(x)
        recon_x = self.decoder(h).view(size=y_size)

        return recon_x
    
    def encode(self, x):
        org_size = x.size()
        y_size = (org_size[0], org_size[1], org_size[2] // 3)
        batch = org_size[0]
        x = x.view(batch, -1)

        h = self.encoder(x)
        return h