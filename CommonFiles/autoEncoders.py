#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch import nn
#from torch.utils import data
#from torch.nn.functional import normalize
#import pandas as pd

class Encoder2(nn.Module):
    def __init__(self,latent_size,input_size,hidden_size_1,hidden_size_2):
        super().__init__()
        
    ### Linear section
        self.encoder_lin=nn.Sequential(
        nn.Linear(input_size, hidden_size_1),
        nn.ReLU(True),
        nn.Linear(hidden_size_1, hidden_size_2),
        nn.ReLU(True), 
        nn.Linear(hidden_size_2, latent_size),
        )
        
    def forward(self,x):
        x=self.encoder_lin(x)
        return x

class Decoder2(nn.Module):
    
    def __init__(self,latent_size,input_size,hidden_size_1,hidden_size_2):
        super().__init__()
        self.decoder_lin=nn.Sequential(
        nn.Linear(latent_size, hidden_size_2),
        nn.ReLU(True),
        nn.Linear(hidden_size_2, hidden_size_1),
        nn.ReLU(True), 
        nn.Linear(hidden_size_1, input_size),
        #nn.Sigmoid(),
        )
        
    def forward(self,x):
        x=self.decoder_lin(x)
        return x

class Encoder3(nn.Module):
    def __init__(self,latent_size,input_size,hidden_size_1,hidden_size_2,hidden_size_3):
        super().__init__()
        
    ### Linear section
        self.encoder_lin=nn.Sequential(
        nn.Linear(input_size, hidden_size_1),
        nn.ReLU(True),
        nn.Linear(hidden_size_1, hidden_size_2),
        nn.ReLU(True), 
        nn.Linear(hidden_size_2, hidden_size_3),
        nn.ReLU(True),
        nn.Linear(hidden_size_3, latent_size),
        )
        
    def forward(self,x):
        x=self.encoder_lin(x)
        return x

class Decoder3(nn.Module):
    
    def __init__(self,latent_size,input_size,hidden_size_1,hidden_size_2,hidden_size_3):
        super().__init__()
        self.decoder_lin=nn.Sequential(
        nn.Linear(latent_size, hidden_size_3),
        nn.ReLU(True), 
        nn.Linear(hidden_size_3, hidden_size_2),
        nn.ReLU(True),
        nn.Linear(hidden_size_2, hidden_size_1),
        nn.ReLU(True), 
        nn.Linear(hidden_size_1, input_size),
        nn.Sigmoid(),
        )
        
    def forward(self,x):
        x=self.decoder_lin(x)
        return x

class Encoder4(nn.Module):
    def __init__(self,latent_size,input_size,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4):
        super().__init__()
        
    ### Linear section
        self.encoder_lin=nn.Sequential(
        nn.Linear(input_size, hidden_size_1),
        nn.ReLU(True),
        nn.Linear(hidden_size_1, hidden_size_2),
        nn.ReLU(True), 
        nn.Linear(hidden_size_2, hidden_size_3),
        nn.ReLU(True),
        nn.Linear(hidden_size_3, hidden_size_4),
        nn.ReLU(True), 
        nn.Linear(hidden_size_4, latent_size),
        )
        
    def forward(self,x):
        x=self.encoder_lin(x)
        return x

class Decoder4(nn.Module):
    
    def __init__(self,latent_size,input_size,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4):
        super().__init__()
        self.decoder_lin=nn.Sequential(
        nn.Linear(latent_size, hidden_size_4),
        nn.ReLU(True),
        nn.Linear(hidden_size_4, hidden_size_3),
        nn.ReLU(True), 
        nn.Linear(hidden_size_3, hidden_size_2),
        nn.ReLU(True),
        nn.Linear(hidden_size_2, hidden_size_1),
        nn.ReLU(True), 
        nn.Linear(hidden_size_1, input_size),
        nn.Sigmoid(),
        )
        
    def forward(self,x):
        x=self.decoder_lin(x)
        return x

def train_epoch_den(encoder,decoder,device,dataloader,loss_fn,optimizer):
    encoder.train()
    decoder.train()
    train_loss=[]
    for item in dataloader: # "_" ignore labels
        encoded_data=encoder(item.float())
        decoded_data=decoder(encoded_data)
        #print(encoded_data[0])
        loss=loss_fn(decoded_data,item.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        #i+=1
    return np.mean(train_loss), encoded_data[0]

def test_epoch_den(encoder,decoder,device,dataloader,loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        conc_out=[]
        conc_label=[]
        for item in dataloader:
            encoded_data=encoder(item.float())
            decoded_data=decoder(encoded_data)
            conc_out.append(decoded_data)
            conc_label.append(item.float())
        conc_out=torch.cat(conc_out)
        conc_label=torch.cat(conc_label)
        test_loss=loss_fn(conc_out,conc_label)
    return test_loss.data, decoded_data, encoded_data

