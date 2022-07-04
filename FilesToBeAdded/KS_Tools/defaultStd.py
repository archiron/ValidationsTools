#!/usr/bin/env python
# coding: utf-8

iDisplay = 9

useTrainLoader = 1
useEncoder = 0
saveEncoder = 1

# xxx parameters
hidden_size_1 = 401 # 400 # 400 #300 # 250 # 78
hidden_size_2 = 201 # 160 # 169 # 80 # 48 # 29
hidden_size_3 = 101 # 80
hidden_size_4 = 51 # 100
useHL3 = 1
useHL4 = 1

N_HL1 = 6 #6
N_HL2 = 2 #5
N_HL3 = 2 #5

#size of latent space
latent_size = 2 # 10 # 9 # 20

#define the optimizer
lr = 0.00001 #learning rate

# define epsilon
epsilon = 1.e-5 ## 1.e-6 : 950 == 1.e-5 : 200

#global parameters
batch_size = 10
nb_epochs = 1000 # 21, 100
percentageTrain = 0.95 ## 0.97 : 950 == 0.95 : 200

############################################
# Layers : 400/160/10 (200)
############################################

