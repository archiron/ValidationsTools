#!/usr/bin/env python
# coding: utf-8

################################################################################
# AEGeneration : create a KS comparison (max diff) between the original curve 
# and the predicted one for different egamma validation releases.
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import time
import os,sys
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.nn.functional import normalize

if len(sys.argv) > 1:
    print(sys.argv)
    print("arg. 0 :", sys.argv[0]) # name of the script
    print("arg. 1 :", sys.argv[1]) # cpu/gpu choice
    cg_pu = sys.argv[1]
else:
    cg_pu = ''
    print("arg. 1 : none.") # cpu/gpu choice

commonPath = '/home/llr/info/chiron_u/PYTHON/ValidationsTools/CommonFiles/'
filePaths = 'pathsLLR.py'
workPath=commonPath[:-12]

print("\nAE Generation")

# Import module
blo_RESULTFOLDER="/data_CMS/cms/chiron/Validations/"
blo_RESULT="/home/llr/info/chiron_u/PYTHON/ValidationsTools/AE_Tools/Results/"
blo_LIB_SOURCE="ChiLib/"
blo_DATA_SOURCE="DATA/"

print('DATA_SOURCE : %s' % blo_DATA_SOURCE)
resultPath = blo_RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo_LIB_SOURCE 
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

tp_1 = 'ElectronMcSignalValidator'
nbFiles = 1000
NB_EVTS = 9000
dfo_folder = 'DEV_03/'
saveEncoder = 1

# xxx parameters
hidden_size_1 = 400
hidden_size_2 = 169

#size of latent space
latent_size = 2 # 10 # 9 # 20

#define the optimizer
lr = 0.00001 #learning rate

# define epsilon
epsilon = 1.e-5 

#global parameters
batch_size = 10
nb_epochs = 100 
percentageTrain = 0.95 

class Encoder2(nn.Module):
    def __init__(self,device,latent_size,input_size,hidden_size_1,hidden_size_2):
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
    
    def __init__(self,device,latent_size,input_size,hidden_size_1,hidden_size_2):
        super().__init__()
        self.decoder_lin=nn.Sequential(
        nn.Linear(latent_size, hidden_size_2),
        nn.ReLU(True),
        nn.Linear(hidden_size_2, hidden_size_1),
        nn.ReLU(True), 
        nn.Linear(hidden_size_1, input_size),
        )
        
    def forward(self,x):
        x=self.decoder_lin(x)
        return x

def train_epoch_den(encoder,decoder,dataloader,loss_fn,optimizer):
    encoder.train()
    decoder.train()
    train_loss=[]
    for item in dataloader: 
        encoded_data=encoder(item)
        decoded_data=decoder(encoded_data)
        loss=loss_fn(decoded_data,item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach())
    train_loss = torch.tensor(train_loss)
    return torch.mean(train_loss), encoded_data[0]

def test_epoch_den(encoder,decoder,dataloader,loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        conc_out=[]
        conc_label=[]
        for item in dataloader:
            encoded_data=encoder(item)
            decoded_data=decoder(encoded_data)
            conc_out.append(decoded_data)
            conc_label.append(item)
        conc_out=torch.cat(conc_out)
        conc_label=torch.cat(conc_label)
        test_loss=loss_fn(conc_out,conc_label)
    return test_loss.data, decoded_data, encoded_data

def createAEfolderName(hs1, hs2, ls): 
    folderName = "/HL_1.{:03d}".format(hs1) + "_HL_2.{:03d}".format(hs2)
    folderName += "_LT.{:02d}".format(ls) + '/' 
    return folderName

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def checkFolder(folder):
    if not os.path.exists(folder): # create folder
        os.makedirs(folder) # create reference folder
    else: # folder already created
        print('%s already created.' % folder)
    return

def getListFiles(path, ext='root'):
    # use getListFiles(str path_where_the_files_are, str 'ext')
    # ext can be root, txt, png, ...
    # default is root
    ext = '.' + ext
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(ext)] # keep only root files
    return onlyfiles

def getBranches(t_p, branchPath):
    b = []
    source = open(branchPath, "r")
    for ligne in source:
        if t_p in ligne:
            tmp = ligne.split(" ", 1)
            b.append(tmp[0].replace(t_p + "/", ""))
    source.close()
    return b

def cleanBranches(branches):
    #if (branches[i] == 'h_ele_seedMask_Tec'): # temp (pbm with nan)
    #if re.search('OfflineV', branches[i]): # temp (pbm with nbins=81 vs nbins=80)
    toBeRemoved = ['h_ele_seedMask_Tec'] # , 'h_ele_convRadius', 'h_ele_PoPtrue_golden_barrel', 'h_ele_PoPtrue_showering_barrel'
    for ele in toBeRemoved:
        if ele in branches:
            branches.remove(ele)

def change_nbFiles(nbFiles_computed, nbFiles):
    if (nbFiles_computed != nbFiles):
        print('the number of computed files (' + '{:d}'.format(nbFiles_computed) + ') is different from the pre supposed number (' + '{:d}'.format(nbFiles) + ').')
        print('switching to the computed number ({:d}).'.format(nbFiles_computed))
        return nbFiles_computed
    else:
        return nbFiles

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

nbBranches = len(branches) 
print('there is {:03d} datasets'.format(nbBranches))

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))

# get list of generated ROOT files
rootPath = "/data_CMS/cms/chiron/ROOT_Files/CMSSW_12_5_0_pre4/"
print('rootPath : {:s}'.format(rootPath))
rootFilesList_0 = getListFiles(rootPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder = resultPath + checkFolderName(dfo_folder)
data_dir = folder + '/{:03d}'.format(nbFiles)
data_res = data_dir + '/AE_RESULTS/'
print('data_dir path : {:s}'.format(data_dir))
print('data_res path : {:s}'.format(data_res))

# get list of text files
pathKSFiles = data_dir
print('KS path : %s' % pathKSFiles)
    
#load data from branchesHistos_NewFiles.txt file ..
fileName = data_dir + "/branchesHistos_NewFiles.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_GPU = True
else:
    device = torch.device("cpu")
    use_GPU = False
if (cg_pu == 'cpu'):
    device = torch.device("cpu")
    use_GPU = False
print('\n===\ndevice : {:s}\n===\n'.format(str(device)))

timeFolder = time.strftime("%Y%m%d-%H%M%S")

folderName = data_res + createAEfolderName(hidden_size_1, hidden_size_2, latent_size) 
checkFolder(folderName)
print('\nComplete folder name : {:s}'.format(folderName))

tic = time.time()

loopMaxValue = 10 # nbBranches 
for i in range(0, loopMaxValue):
    t1 = time.time() # time inside loop
    print('{:s}\n'.format(branches[i]))
    df = []
    fileName = resultPath + "/histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
    if Path(fileName).exists():
        print('{:s} exist'.format(fileName))
        df = pd.read_csv(fileName)
    else:
        print('{:s} does not exist'.format(fileName))
        continue

    # add a subfolder with the name of the histo and a folder with date/time
    folderNameBranch = folderName + branches[i] + '/' + timeFolder
    checkFolder(folderNameBranch)
    print('\n===== folderNameBranch : {:s} ====='.format(folderNameBranch))

    df_entries = []
    linOp = []

    for line in Lines:
        rel,b = line.rstrip().split(',', 1)
        hName = b.rstrip().split(',', 1)[0]
        if ( str(hName) == str(branches[i])):
            linOp.append(line)

    torch_tensor_entries = []
    torch_tensor_entries_n = []

    train_loader = []
    test_loader = []

    # add a subfolder for the losses
    folderNameLosses = folderNameBranch + '/Losses/'
    checkFolder(folderNameLosses)
    print('\nfolderNameLosses : {:s}'.format(folderNameLosses))

    tmp = df 
    cols = df.columns.values
    cols_entries = cols[6::2]
    df_entries = tmp[cols_entries]
    (_, Ncols) = df_entries.shape

    # get nb of columns & rows for histos & remove over/underflow
    (Nrows, Ncols) = df_entries.shape
    df_entries = df_entries.iloc[:, 1:Ncols-1]
    (Nrows, Ncols) = df_entries.shape
    print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branches[i]))
    print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branches[i]))

    # add a subfolder for the losses
    folderNameLoader = folderNameBranch + '/TrainTestLOADER/'
    checkFolder(folderNameLoader)
    print('\nfolderNameLoader : {:s}'.format(folderNameLoader))

    trainName = folderNameLoader + "multi_train_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    testName = folderNameLoader + "multi_test_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"

    # creating torch tensor from df_entries
    torch_tensor_entries = torch.tensor(df_entries.values, device=device).float() # 
    # normalize the tensor
    torch_tensor_entries_n = normalize(torch_tensor_entries, p=2.0)

    train_size=int(percentageTrain*len(torch_tensor_entries)) 
    test_size=len(torch_tensor_entries)-train_size
    print('{:d} : [train size, test size] : [{:d}, {:d}]'.format(i,train_size, test_size))
    train_tmp, test_tmp = data.random_split(torch_tensor_entries_n,[train_size,test_size])

    train_loader = data.DataLoader(train_tmp,batch_size=batch_size)
    test_loader = data.DataLoader(test_tmp,batch_size=batch_size)
    
    torch.save(train_loader,trainName)
    torch.save(test_loader,testName)
    
    # load all data to device for gpu
    if use_GPU:
        loss_fn = torch.nn.MSELoss().cuda()
        for item in train_loader: 
            item.to(device)
        for item in test_loader:
            item.to(device)
    else:
        loss_fn=torch.nn.MSELoss()

    #define the network
    encoder=Encoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
    decoder=Decoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
    nbLayer = 2

    encoder.to(device)
    decoder.to(device)

    params_to_optimize=[
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
    ]

    optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)
    history_da={'train_loss':[],'test_loss':[]}
    L_out = []
    LatentValues_Train = []
    LatentValues_Test = []

    # Ready for calculation
    encoderName = folderNameLoader + "/mono_encoder_{:01d}_".format(nbLayer) + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    decoderName = folderNameLoader + "/mono_decoder_{:01d}_".format(nbLayer) + branches[i] + "_{:03d}".format(nbFiles) + ".pth"

    t_train1 = time.time() # time for train
    for epoch in range(nb_epochs):
        #t_epoch1 = time.time() # time for epoch
        train_loss, encoded_out=train_epoch_den(encoder=encoder, decoder=decoder,
            dataloader=train_loader, loss_fn=loss_fn,optimizer=optim) # 
        
        test_loss, d_out, latent_out=test_epoch_den(encoder=encoder, decoder=decoder,
            dataloader=test_loader, loss_fn=loss_fn) # 
        L_out.append(d_out)
        LatentValues_Train.append(encoded_out)
        LatentValues_Test.append(latent_out)
        history_da['train_loss'].append(train_loss)
        history_da['test_loss'].append(test_loss)
        #t_epoch2 = time.time() # time for epoch
        #print('epoch {:03d} done in {:.4f} s. for histo : {:s}'.format(epoch, (t_epoch2-t_epoch1), branches[i]))
        
        bo1 = train_loss < epsilon 
        bo2 = test_loss < epsilon
        if (bo1 and bo2):
            break

    t_train2 = time.time() # time for train
    print('Train done in {:.4f} seconds for histo : {:s}'.format((t_train2-t_train1), branches[i]))
    #print('epoch : %03d : tr_lo = %e : te_lo = %e' % (epoch, train_loss, test_loss))
    if ( saveEncoder == 1 ): # warning encoder & decoder are needed for next computations
        torch.save(encoder,encoderName)
        torch.save(decoder,decoderName)

    labels_Train = []
    labels_Test = []
    x_Train = []
    y_Train = []
    x_Test = []
    y_Test = []
    for ind in range(0, len(LatentValues_Train)):
        x_Train.append(LatentValues_Train[ind][0])
        y_Train.append(LatentValues_Train[ind][1])
        labels_Train.append(i)
    
    for ind in range(0, len(LatentValues_Test)):
        x_Test.append(LatentValues_Test[ind][0][0])
        y_Test.append(LatentValues_Test[ind][0][1])
        labels_Test.append(i)

    # Ready for prediction
    print('using %s\n' % encoderName)

    lossesVal = []
    latentVal = []
    for elem in linOp:
        print('linOp : ', elem.split(',')[0])
        rel, hName,line = elem.rstrip().split(',', 2)
        new = line.rstrip().split(',')
        new = np.asarray(new).astype(float)
        df_new = pd.DataFrame(new).T # otherwise, one column with 50 rows instead of 1 line with 50 columns

        # creating torch tensor from df_entries/errors
        torch_tensor_new = torch.tensor(df_new.values,device=device).float()
        # normalize the tensor
        torch_tensor_entries_n = normalize(torch_tensor_new, p=2.0)
        test_loader_n = data.DataLoader(torch_tensor_entries_n)
        # load all data to device
        if use_GPU:
            loss_fn = torch.nn.MSELoss().cuda()
            for item in test_loader_n:
                item.to(device)
        else:
            loss_fn=torch.nn.MSELoss()

        encoder = torch.load(encoderName)
        decoder = torch.load(decoderName)
        encoder.to(device)
        decoder.to(device)

        params_to_optimize=[
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
        ]

        optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)

        # Forward pass: Compute predicted y by passing x to the model
        new_loss, y_pred_new, latent_out = test_epoch_den(encoder=encoder,
                decoder=decoder, dataloader=test_loader_n, loss_fn=loss_fn)

        # Compute and print loss
        lossesVal.append([rel,new_loss.item()])
        latentVal.append(latent_out[0])#.cpu().numpy()
    t2 = time.time() # time inside loop
    print('Done in {:.4f} seconds for histo : {:s}'.format((t2-t1), branches[i]))

toc = time.time()
print('Done in {:.4f} seconds for device : {:s}'.format((toc-tic), str(device)))
print('end')

