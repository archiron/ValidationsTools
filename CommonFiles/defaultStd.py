#!/usr/bin/env python
# coding: utf-8

useTrainLoader = 0
useEncoder = 0
saveEncoder = 1
TimeFolderRef='20221220-160319'

# xxx parameters
hidden_size_1 = 401 # 401 # 400 # 400 #300 # 250 # 78
hidden_size_2 = 169 # 201 # 160 # 169 # 80 # 48 # 29
hidden_size_3 = 169 # 80
hidden_size_4 = 51 # 100
hidden_size_5 = 16 
hidden_size_6 = 8 
hidden_size_7 = 417 

useHL1 = 1 # always 1
useHL2 = 1 # always 1
useHL3 = 1
useHL4 = 0
useHL5 = 0
useHL6 = 0
useHL7 = 0

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
nb_epochs = 100 # 21, 100
percentageTrain = 0.95 ## 0.97 : 950 == 0.95 : 200

############################################
# Layers : 400/160/10 (200)
#          401/201/101/51 (950)
############################################

