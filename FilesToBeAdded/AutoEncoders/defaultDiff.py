#!/usr/bin/env python
# coding: utf-8

iDisplay = 0

#nbFiles = 20
nbFiles = 200
#nbFiles = 950

useTrainLoader = 1
useEncoder = 0

# xxx parameters
hidden_size_1 = 300 #300 250
hidden_size_2 = 150 # 80 48
hidden_size_3 = 80
hidden_size_4 = 40

#size of latent space
latent_size =  20 # 10 20

#define the optimizer
lr = 0.00001 #learning rate

# define epsilon
epsilon = 1.e-5

#global parameters
batch_size = 300
nb_epochs = 100 # 21, 100
percentageTrain = 0.97

############################################
# 0 : 600/300/20 (200)
# 1 : 400/160/10 (200)
# 2 : 600/300/20 (200) pas bon
# 3 : 400/160/10 (200)
# 4 : 400/160/10 (200)
# 5 : 400/160/10 (200)
# 6 : 400/160/10 (200)
# 7 : 400/160/10 (200)
############################################

