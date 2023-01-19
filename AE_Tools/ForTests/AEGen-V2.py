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

import datetime, time
import sys
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("AEGen V2 - arg. 0 :", sys.argv[0]) # name of the script
    print("AEGen V2 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("AEGen V2 - arg. 2 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.nn.functional import normalize

print("\nAE Gen-V2")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE 
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicAutoEncoderFunctions import *

from DecisionBox import *
DB = DecisionBox()

useTrainLoader = 0
useEncoder = 0
saveEncoder = 1
TimeFolderRef='20221220-160319'

# xxx parameters
hidden_size_1 = 400 # 401 # 400 # 400 #300 # 250 # 78
hidden_size_2 = 169 # 201 # 160 # 169 # 80 # 48 # 29
hidden_size_3 = 10 # 80
hidden_size_4 = 51 # 100
useHL3 = 0
useHL4 = 0

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
nb_epochs = 5 # 21, 100
percentageTrain = 0.95 ## 0.97 : 950 == 0.95 : 200

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
        #nn.Sigmoid(),
        )
        
    def forward(self,x):
        x=self.decoder_lin(x)
        return x

def createAutoEncoderRef(nbFiles, nbBranches, device, lr, epsilon, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, latent_size, batch_size, nb_epochs, percentageTrain):
    Text = []
    Text.append(' <h1><center><b><font color=\'blue\'>AutoEncoder with {:d} files.</font></b></center></h1> <br>\n'.format(nbFiles))
    Text.append('<table border=\"1\" cellpadding=\"5\" width=\"100%\">\n')
    Text.append('<tr>\n')
    Text.append('<td>\n')
    Text.append('nbFiles = ' + str(nbFiles) + '<br>\n')
    Text.append('there is {:03d} datasets<br>\n'.format(nbBranches))
    Text.append('device : {:s}<br>\n'.format(str(device)))
    Text.append('</td>\n')
    Text.append('<td>\n')
    Text.append('learning rate lr = ' + str(lr) + '<br>\n')
    Text.append('epsilon = ' + str(epsilon) + '<br>\n')
    Text.append('</td>\n')
    Text.append('<td>\n')
    Text.append('hidden_size_1 = ' + str(hidden_size_1) + '<br>\n')
    Text.append('hidden_size_2 = ' + str(hidden_size_2) + '<br>\n')
    if useHL3 == 1:
        Text.append('hidden_size_3 = ' + str(hidden_size_3) + '<br>\n')
    if useHL4 == 1:
        Text.append('hidden_size_4 = ' + str(hidden_size_4) + '<br>\n')
    Text.append('latent_size = ' + str(latent_size) + '<br>\n')
    Text.append('</td>\n')
    Text.append('<td>\n')
    Text.append('batch_size = ' + str(batch_size) + '<br>\n')
    Text.append('nb_epochs = ' + str(nb_epochs) + '<br>\n')
    Text.append('percentageTrain = ' + str(percentageTrain) + '<br>\n')
    Text.append('</td>\n')
    Text.append('</tr>\n')
    Text.append('</table>\n')
    Text.append('<br>\n')
    return Text

def train_epoch_den(encoder,decoder,device,dataloader,loss_fn,optimizer):
    encoder.train()
    decoder.train()
    train_loss=[]
    for item in dataloader: # "_" ignore labels
        #item.to(device)
        encoded_data=encoder(item) # .float()
        decoded_data=decoder(encoded_data)
        loss=loss_fn(decoded_data,item) # .float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach()) # .cpu().numpy()
    train_loss = torch.tensor(train_loss) # .clone().detach()
    return torch.mean(train_loss), encoded_data[0]
    #return np.mean(train_loss), encoded_data[0]

def test_epoch_den(encoder,decoder,device,dataloader,loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        conc_out=[]
        conc_label=[]
        for item in dataloader:
            #item.to(device)
            encoded_data=encoder(item) # .float()
            decoded_data=decoder(encoded_data)
            conc_out.append(decoded_data)
            conc_label.append(item) # .float()
        conc_out=torch.cat(conc_out)
        conc_label=torch.cat(conc_label)
        test_loss=loss_fn(conc_out,conc_label)
    return test_loss.data, decoded_data, encoded_data

def createAEfolderName(hs1, hs2, hs3, hs4, useHL3, useHL4, ls): # , tF, nbFiles, histoName
    folderName = "/HL_1.{:03d}".format(hs1) + "_HL_2.{:03d}".format(hs2)
    if useHL3 == 1:
        folderName += "_HL_3.{:03d}".format(hs3)
    if useHL4 == 1:
        folderName += "_HL_4.{:03d}".format(hs4)
    folderName += "_LT.{:02d}".format(ls) + '/' # + "{:03d}".format(nbFiles)
    #folderName += '/' + histoName + '/' # tF + 
    return folderName

arrayKSValues = []
rels = []

y_pred_n = []
y_pred_o = []

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
    
nbBranches = len(branches) # [0:8]
print('there is {:03d} datasets'.format(nbBranches))

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))

# get list of generated ROOT files
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder = resultPath + checkFolderName(dfo.folder)
data_dir = folder + '/{:03d}'.format(nbFiles)
print('data_dir path : {:s}'.format(data_dir))
#data_res = data_dir + '/AE_RESULTS/'
data_res = '/pbs/home/c/chiron/public/TEMP/AE_RESULTS/'
print('data_res path : {:s}'.format(data_res))

# get list of added ROOT files
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList)) + ' added ROOT files')
for item in rootFilesList:
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
for elem in sortedRels:
    print(elem)

# get list of text files
pathKSFiles = data_dir
print('KS path : %s' % pathKSFiles)
KSlistFiles = []
tmp = getListFiles(pathKSFiles, 'txt')
for elem in tmp:
    if (elem[5:10] == '_diff'): # to keep only histo_differences_KScurves files
        KSlistFiles.append(elem)
print(KSlistFiles, len(KSlistFiles))
    
for item in KSlistFiles:
    print('file : %s' % item)
    aa = item.split('__')[0]
    fileName = pathKSFiles + '/' + item
    file1 = open(fileName, 'r')
    bb = file1.readlines()
    for elem in bb:
        tmp = []
        cc = elem.split(' : ')
        tmp = [cc[0], aa, float(cc[1][:-1])]
        arrayKSValues.append(tmp)
sortedArrayKSValues = sorted(arrayKSValues, key = lambda x: x[0]) # gives an array with releases sorted
for elem in sortedArrayKSValues:
    print("sortedArrayKSValues", elem)

if (len(KSlistFiles) != len(rootFilesList)):
    print('you must have the same number of KS files than releases')
    exit()
else:
    print('we have the same number of KS files than releases')

#load data from branchesHistos_NewFiles.txt file ..
fileName = data_dir + "/branchesHistos_NewFiles.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()

device = torch.device("cpu")
print('\n===\ndevice : {:s}\n===\n'.format(str(device)))


t = datetime.datetime.today()
timeFolder = time.strftime("%Y%m%d-%H%M%S")

folderName = data_res + createAEfolderName(hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, useHL3, useHL4, latent_size) # , timeFolder, nbFiles, branch
checkFolder(folderName)
print('\nComplete folder name : {:s}'.format(folderName))

# export parameters of the layers
exportParameters = folderName + '/parameters.html'
print(exportParameters)
fParam = open(exportParameters, 'w')  # html page
for line in createAutoEncoderRef(nbFiles, nbBranches, device, lr, epsilon, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, latent_size, batch_size, nb_epochs, percentageTrain):
    fParam.write(line)
fParam.close()

loopMaxValue = 6 #nbBranches #25 # nbBranches
reste = loopMaxValue % 2
if (reste == 1):
    loopMax = loopMaxValue - 1
else : # reste = 0
    loopMax = loopMaxValue
loopInit = 0

print('\nLoop 2')
time11 = time.time()
for i in range(loopInit, loopMax, 2):
    t_branch11 = time.time()
    branch1 = branches[i]
    branch2 = branches[i+1]
    print('{:s}-{:s}\n'.format(branch1, branch2))
    df1 = []
    df2 = []
    fileName = resultPath + "/histo_" + branch1 + '_{:03d}'.format(nbFiles) + ".txt"
    if Path(fileName).exists():
        print('{:s} exist'.format(fileName))
        df1 = pd.read_csv(fileName)
    else:
        print('{:s} does not exist'.format(fileName))
        continue
    fileName = resultPath + "/histo_" + branch2 + '_{:03d}'.format(nbFiles) + ".txt"
    if Path(fileName).exists():
        print('{:s} exist'.format(fileName))
        df2 = pd.read_csv(fileName)
    else:
        print('{:s} does not exist'.format(fileName))
        continue

    folderNameBranch1 = folderName + branch1 + '/' + timeFolder
    folderNameBranch2 = folderName + branch2 + '/' + timeFolder
    checkFolder(folderNameBranch1)
    checkFolder(folderNameBranch2)
    print('\n===== folderNameBranch1 : {:s} ====='.format(folderNameBranch1))
    print('\n===== folderNameBranch2 : {:s} ====='.format(folderNameBranch2))

    resumeHisto1 = folderNameBranch1 + '/histo_' + '{:s}'.format(str(branch1))
    resumeHisto1 += '.html'
    resumeHisto2 = folderNameBranch2 + '/histo_' + '{:s}'.format(str(branch2))
    resumeHisto2 += '.html'
    print(resumeHisto1, resumeHisto2)
    fHisto1 = open(resumeHisto1, 'w')  # web page
    fHisto1.write("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n")
    fHisto1.write("<html>\n")
    fHisto1.write("<head>\n")
    fHisto1.write("<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n")
    fHisto1.write("<title> Resume of ZEE_14 predictions"+ str(branch1)+ " </title>\n")  # 
    fHisto1.write("</head>\n")
    fHisto2 = open(resumeHisto2, 'w')  # web page
    fHisto2.write("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n")
    fHisto2.write("<html>\n")
    fHisto2.write("<head>\n")
    fHisto2.write("<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n")
    fHisto2.write("<title> Resume of ZEE_14 predictions"+ str(branch2)+ " </title>\n")  # 
    fHisto2.write("</head>\n")

    fHisto1.write(' <h1><center><b><font color=\'blue\'>{:s}</font></b></center></h1> <br>\n'.format(str(branch1)))
    fHisto1.write('<b>folderName : </b>{:s}<br>\n'.format(folderNameBranch1))
    fHisto1.write('<br>\n')
    fHisto2.write(' <h1><center><b><font color=\'blue\'>{:s}</font></b></center></h1> <br>\n'.format(str(branch2)))
    fHisto2.write('<b>folderName : </b>{:s}<br>\n'.format(folderNameBranch2))
    fHisto2.write('<br>\n')
    
    df_entries1 = []
    df_entries2 = []

    torch_tensor_entries1 = []
    torch_tensor_entries_n1 = []
    torch_tensor_entries2 = []
    torch_tensor_entries_n2 = []

    train_loader1 = []
    train_loader2 = []
    test_loader1 = []
    test_loader2 = []

    # add a subfolder for the losses
    folderNameLosses1 = folderNameBranch1 + '/Losses/'
    checkFolder(folderNameLosses1)
    print('\nfolderNameLosses1 : {:s}'.format(folderNameLosses1))
    folderNameLosses2 = folderNameBranch2 + '/Losses/'
    checkFolder(folderNameLosses2)
    print('\nfolderNameLosses2 : {:s}'.format(folderNameLosses2))

    lossesValues1 = folderNameLosses1 + "/lossesValues_" + branch1 + ".txt"
    print("loss values file 1 : %s\n" % lossesValues1)
    wLoss1 = open(lossesValues1, 'w')
    lossesValues2 = folderNameLosses2 + "/lossesValues_" + branch2 + ".txt"
    print("loss values file 2 : %s\n" % lossesValues2)
    wLoss2 = open(lossesValues2, 'w')

    tmp1 = df1 
    cols1 = df1.columns.values
    cols_entries1 = cols1[6::2]
    df_entries1 = tmp1[cols_entries1]
    (_, Ncols1) = df_entries1.shape
    tmp2 = df2 
    cols2 = df2.columns.values
    cols_entries2 = cols2[6::2]
    df_entries2 = tmp2[cols_entries2]
    (_, Ncols2) = df_entries2.shape

    # get nb of columns & rows for histos & remove over/underflow
    (Nrows1, Ncols1) = df_entries1.shape
    print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows1, Ncols1, branch1))
    df_entries1 = df_entries1.iloc[:, 1:Ncols1-1]
    (Nrows1, Ncols1) = df_entries1.shape
    print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows1, Ncols1, branch1))
    fHisto1.write('nb of columns for histo {:s} after extraction : [{:3d}, {:3d}]<br>\n'.format(branch1, Nrows1, Ncols1))
    (Nrows2, Ncols2) = df_entries2.shape
    print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows2, Ncols2, branch2))
    df_entries2 = df_entries2.iloc[:, 1:Ncols2-1]
    (Nrows2, Ncols2) = df_entries2.shape
    print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows2, Ncols2, branch2))
    fHisto2.write('nb of columns for histo {:s} after extraction : [{:3d}, {:3d}]<br>\n'.format(branch2, Nrows2, Ncols2))

    fHisto1.write('<br>\n')
    fHisto2.write('<br>\n')

    # add a subfolder for the losses
    folderNameLoader1 = folderNameBranch1 + '/TrainTestLOADER/'
    checkFolder(folderNameLoader1)
    print('\nfolderNameLoader : {:s}'.format(folderNameLoader1))
    folderNameLoader2 = folderNameBranch2 + '/TrainTestLOADER/'
    checkFolder(folderNameLoader2)
    print('\nfolderNameLoader : {:s}'.format(folderNameLoader2))

    trainName1 = folderNameLoader1 + "multi_train_loader_" + branch1 + "_{:03d}".format(nbFiles) + ".pth"
    testName1 = folderNameLoader1 + "multi_test_loader_" + branch1 + "_{:03d}".format(nbFiles) + ".pth"
    trainName2 = folderNameLoader2 + "multi_train_loader_" + branch2 + "_{:03d}".format(nbFiles) + ".pth"
    testName2 = folderNameLoader2 + "multi_test_loader_" + branch2 + "_{:03d}".format(nbFiles) + ".pth"

    fHisto1.write('creating train[test]_loader<br>\n')
    fHisto1.write('creating : {:s} OK.<br>\n'.format(trainName1))
    fHisto1.write('creating : {:s} OK.<br>\n'.format(testName1))
    fHisto1.write('<br>\n')
    fHisto2.write('creating train[test]_loader<br>\n')
    fHisto2.write('creating : {:s} OK.<br>\n'.format(trainName2))
    fHisto2.write('creating : {:s} OK.<br>\n'.format(testName2))
    fHisto2.write('<br>\n')
    # creating torch tensor from df_entries
    torch_tensor_entries1 = torch.tensor(df_entries1.values, device=device).float()
    torch_tensor_entries2 = torch.tensor(df_entries2.values, device=device).float()
    print('max df')
    print(df_entries1.values.max())
    print(df_entries2.values.max())
    MAXMax1 = df_entries1.values.max()
    MAXMax2 = df_entries2.values.max()
    print('MAXMax1 : %e' % MAXMax1)
    print('MAXMax2 : %e' % MAXMax2)
    if (MAXMax1 == 1.e38):
        print(colorText('ATTENTION, Kolossal PBM !!!', 'blue'))
    if (MAXMax2 == 1.e38):
        print(colorText('ATTENTION, Kolossal PBM !!!', 'blue'))
    # normalize the tensor
    torch_tensor_entries_n1 = normalize(torch_tensor_entries1, p=2.0)
    torch_tensor_entries_n2 = normalize(torch_tensor_entries2, p=2.0)
    print(torch_tensor_entries1.shape) # OK
    print(torch_tensor_entries2.shape) # OK

    train_size1  = int(percentageTrain*len(torch_tensor_entries1)) # 
    test_size1 = len(torch_tensor_entries1)-train_size1
    train_size2 = int(percentageTrain*len(torch_tensor_entries2)) # 
    test_size2 = len(torch_tensor_entries2)-train_size2
    print('%d : train size 1 : %d' % (i,train_size1))
    print('%d : test size  1 : %d' % (i,test_size1))
    print('%d : train size 2 : %d' % (i+1,train_size2))
    print('%d : test size  2 : %d' % (i+1,test_size2))
    fHisto1.write('train size 1 : {:d}<br>\n'.format(train_size1))
    fHisto1.write('test size 1  : {:d}<br>\n'.format(test_size1))
    fHisto2.write('train size 2 : {:d}<br>\n'.format(train_size2))
    fHisto2.write('test size 2  : {:d}<br>\n'.format(test_size2))
    train_tmp1, test_tmp1 = data.random_split(torch_tensor_entries_n1,[train_size1,test_size1])
    train_tmp2, test_tmp2 = data.random_split(torch_tensor_entries_n2,[train_size2,test_size2])

    train_loader1 = data.DataLoader(train_tmp1,batch_size=batch_size)
    test_loader1 = data.DataLoader(test_tmp1,batch_size=batch_size)
    train_loader2 = data.DataLoader(train_tmp2,batch_size=batch_size)
    test_loader2 = data.DataLoader(test_tmp2,batch_size=batch_size)

    print('saving ... %s-%s' % (trainName1,trainName2))
    torch.save(train_loader1,trainName1)
    torch.save(test_loader1,testName1)
    torch.save(train_loader2,trainName2)
    torch.save(test_loader2,testName2)
    print('save OK.\n')
    fHisto1.write('save : {:s} OK.<br>\n'.format(trainName1))
    fHisto1.write('save : {:s} OK.<br>\n'.format(testName1))
    fHisto2.write('save : {:s} OK.<br>\n'.format(trainName2))
    fHisto2.write('save : {:s} OK.<br>\n'.format(testName2))
    fHisto1.write('<br>\n')
    fHisto2.write('<br>\n')

    # load all data to device for gpu
    loss_fn1=torch.nn.MSELoss()

    #define the network
    fHisto1.write('define the network (encoder/decoder)<br>\n')
    fHisto2.write('define the network (encoder/decoder)<br>\n')
    encoder1=Encoder2(device,latent_size,Ncols1,hidden_size_1,hidden_size_2)
    decoder1=Decoder2(device,latent_size,Ncols1,hidden_size_1,hidden_size_2)
    fHisto1.write('using <b>2</b> layers encoder/decoder<br>\n')
    fHisto2.write('using <b>2</b> layers encoder/decoder<br>\n')
    nbLayer = 2

    encoder1.to(device)
    decoder1.to(device)

    params_to_optimize1=[
    {'params': encoder1.parameters()},
    {'params': decoder1.parameters()}
    ]

    optim1=torch.optim.Adam(params_to_optimize1,lr=lr,weight_decay=1e-05)
    history_da1={'train_loss':[],'test_loss':[]}
    history_da2={'train_loss':[],'test_loss':[]}
    L_out1 = []
    L_out2 = []
    LatentValues_Train1 = []
    LatentValues_Train2 = []
    LatentValues_Test1 = []
    LatentValues_Test2 = []

    # Ready for calculation
    encoderName1 = folderNameLoader1 + "/mono_encoder_{:01d}_".format(nbLayer) + branch1 + "_{:03d}".format(nbFiles) + ".pth"
    decoderName1 = folderNameLoader1 + "/mono_decoder_{:01d}_".format(nbLayer) + branch1 + "_{:03d}".format(nbFiles) + ".pth"
    encoderName2 = folderNameLoader2 + "/mono_encoder_{:01d}_".format(nbLayer) + branch2 + "_{:03d}".format(nbFiles) + ".pth"
    decoderName2 = folderNameLoader2 + "/mono_decoder_{:01d}_".format(nbLayer) + branch2 + "_{:03d}".format(nbFiles) + ".pth"
    fHisto1.write('encoderName : {:s}.<br>\n'.format(encoderName1))
    fHisto1.write('decoderName : {:s}.<br>\n'.format(decoderName1))
    fHisto1.write('<br>\n')
    fHisto2.write('encoderName : {:s}.<br>\n'.format(encoderName2))
    fHisto2.write('decoderName : {:s}.<br>\n'.format(decoderName2))
    fHisto2.write('<br>\n')

    # add a subfolder for the pictures
    folderNamePict1 = folderNameBranch1 + '/Pictures/'
    folderNamePict2 = folderNameBranch2 + '/Pictures/'
    checkFolder(folderNamePict1)
    checkFolder(folderNamePict2)
    print('\nfolderNamePict : {:s}-{:s}'.format(folderNamePict1,folderNamePict2))

    lossesPictureName1 = folderNamePict1 + '/loss_plots_' + branch1 + "_{:03d}".format(nbFiles) + '_V2.png'
    lossesPictureName2 = folderNamePict2 + '/loss_plots_' + branch2 + "_{:03d}".format(nbFiles) + '_V2.png'
    fHisto1.write('Calculating encoder/decoder<br>\n')
    fHisto2.write('Calculating encoder/decoder<br>\n')
    for epoch1 in range(nb_epochs):
        #print('epoch : {:02d}'.format(epoch1))
        train_loss1, encoded_out1 = train_epoch_den(encoder=encoder1, decoder=decoder1,device=device,
            dataloader=train_loader1, loss_fn=loss_fn1,optimizer=optim1)
        test_loss1, d_out1, latent_out1 = test_epoch_den(encoder=encoder1, decoder=decoder1,device=device,
            dataloader=test_loader1, loss_fn=loss_fn1)
        L_out1.append(d_out1)
        LatentValues_Train1.append(encoded_out1) # .detach().numpy()
        LatentValues_Test1.append(latent_out1)
        history_da1['train_loss'].append(train_loss1)
        history_da1['test_loss'].append(test_loss1)
    
    # load all data to device for gpu
    loss_fn2=torch.nn.MSELoss()

    encoder2=Encoder2(device,latent_size,Ncols2,hidden_size_1,hidden_size_2)
    decoder2=Decoder2(device,latent_size,Ncols2,hidden_size_1,hidden_size_2)
    encoder2.to(device)
    decoder2.to(device)
    
    params_to_optimize2=[
    {'params': encoder2.parameters()},
    {'params': decoder2.parameters()}
    ]
    optim2=torch.optim.Adam(params_to_optimize2,lr=lr,weight_decay=1e-05)
    
    for epoch2 in range(nb_epochs):
        #print('epoch : {:02d}'.format(epoch2))
        train_loss2, encoded_out2 = train_epoch_den(encoder=encoder2, decoder=decoder2,device=device,
            dataloader=train_loader2, loss_fn=loss_fn2,optimizer=optim2)
        test_loss2, d_out2, latent_out2 = test_epoch_den(encoder=encoder2, decoder=decoder2,device=device,
            dataloader=test_loader2, loss_fn=loss_fn2)
        L_out2.append(d_out2)
        LatentValues_Train2.append(encoded_out2) # .detach().numpy()
        LatentValues_Test2.append(latent_out2)
        history_da2['train_loss'].append(train_loss2)
        history_da2['test_loss'].append(test_loss2)

    #r1 = (train_loss1 - test_loss1) / (train_loss1 + test_loss1)
    #r2 = (train_loss2 - test_loss2) / (train_loss2 + test_loss2)
    #print('epoch : %03d : tr_lo = %e : te_lo = %e : r1 = %e' % (epoch1, train_loss1, test_loss1, r1))
    #print('epoch : %03d : tr_lo = %e : te_lo = %e : r2 = %e' % (epoch2, train_loss2, test_loss2, r2))
    #fHisto1.write('epoch : {:03d} : train_loss = {:e} : test_loss = {:e}<br>\n'.format(epoch1, train_loss1, test_loss2))
    #fHisto2.write('epoch : {:03d} : train_loss = {:e} : test_loss = {:e}<br>\n'.format(epoch2, train_loss2, test_loss2))
    if ( saveEncoder == 1 ): # warning encoder & decoder are needed for next computations
        torch.save(encoder1,encoderName1)
        torch.save(decoder1,decoderName1)
        torch.save(encoder2,encoderName2)
        torch.save(decoder2,decoderName2)
        fHisto1.write('save : {:s} OK.<br>\n'.format(encoderName1))
        fHisto1.write('save : {:s} OK.<br>\n'.format(decoderName1))
        fHisto2.write('save : {:s} OK.<br>\n'.format(encoderName2))
        fHisto2.write('save : {:s} OK.<br>\n'.format(decoderName2))
    fHisto1.write('<br>\n')
    fHisto2.write('<br>\n')

    wLoss1.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
                % (hidden_size_1, hidden_size_2, latent_size, train_loss1, test_loss1))
    wLoss2.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
                % (hidden_size_1, hidden_size_2, latent_size, train_loss2, test_loss2))

    createLossPictures(branch1, history_da1, epoch1+1, lossesPictureName1)
    createLossPictures(branch2, history_da2, epoch2+1, lossesPictureName2)
    wLoss1.close()
    wLoss2.close()
    t_branch21 = time.time()
    print('time for branches {:s}/{:s} : {:f}'.format(branch1, branch2, t_branch21 - t_branch11))

if (reste == 1) :
    print("a faire")
fHisto1.close()
fHisto2.close()
time21 = time.time()



print('\n Recap ')
print('time for loop 2 : {:f}'.format(time21 - time11))
print('end')

