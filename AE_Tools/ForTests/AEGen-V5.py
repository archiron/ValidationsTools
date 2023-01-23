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
    print("AEGen V1 - arg. 0 :", sys.argv[0]) # name of the script
    print("AEGen V1 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("AEGen V1 - arg. 2 :", sys.argv[2]) # FileName for paths
    print("AEGen V1 - arg. 3 :", sys.argv[3]) # nb of datasets
    print("AEGen V1 - arg. 4 :", sys.argv[4]) # dataset name
    print("AEGen V1 - arg. 5 :", sys.argv[5]) # odd / even - cpu/gpu option
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
    nbBranches = int(sys.argv[3])
    branch = sys.argv[4][1:]
    print('branch : {:s}'.format(branch))
    evenOdd = sys.argv[5]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.nn.functional import normalize

print("\nAE Gen-V1")

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
#branches = []
#source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
#branches = getBranches(tp_1, source)
#cleanBranches(branches) # remove some histo wich have a pbm with KS.

#nbBranches = len(branches) # [0:8]
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

# si evenOdd = even (pair) : on ne fait que la boucle paire
## si reste = 1 (loopMaxValue impaire) alors loopMax = loopMaxValue
## si reste = 0 (loopMaxValue impaire) alors loopMax = loopMaxValue + 1
# si evenOdd = odd (impair) : on ne fait que la boucle impaire
## si reste = 1 (loopMaxValue impaire) alors loopMax = loopMaxValue + 1
## si reste = 0 (loopMaxValue impaire) alors loopMax = loopMaxValue
# si evenOdd = 'both' alors boucle normale : loopMax = loopMaxValue
loopMaxValue = 6 #nbBranches #25 # nbBranches
loopInit = 0
loopStep = 2
loopMax = loopMaxValue
reste = loopMaxValue % 2
if ((reste == 1) and (evenOdd == 'odd')):
    loopMax = loopMaxValue + 1
elif ((reste == 0) and (evenOdd == 'even')):
    loopMax = loopMaxValue + 1
if (evenOdd == 'even'):
    loopInit = 0
else :
    loopInit = 1
if (evenOdd == 'both'):
    loopStep = 1
    loopInit = 0

time1 = time.time()
#for i in range(loopInit, loopMax,loopStep):
#branch = branches[i]
print('{:s}\n'.format(branch))
df = []
fileName = resultPath + "/histo_" + branch + '_{:03d}'.format(nbFiles) + ".txt"
if Path(fileName).exists():
    print('{:s} exist'.format(fileName))
    df = pd.read_csv(fileName)
else:
    print('{:s} does not exist'.format(fileName))
    exit()

folderNameBranch = folderName + branch + '/' + timeFolder
checkFolder(folderNameBranch)
print('\n===== folderNameBranch : {:s} ====='.format(folderNameBranch))

resumeHisto = folderNameBranch + '/histo_' + '{:s}'.format(str(branch))
resumeHisto += '.html'
print(resumeHisto)
textHisto = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
textHisto += "<html>\n"
textHisto += "<head>\n"
textHisto += "<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n"
textHisto += "<title> Resume of ZEE_14 predictions"+ str(branch)+ " </title>\n"  # 
textHisto += "</head>\n"

textHisto += ' <h1><center><b><font color=\'blue\'>{:s}</font></b></center></h1> <br>\n'.format(str(branch))
textHisto += '<b>folderName : </b>{:s}<br>\n'.format(folderNameBranch)
textHisto += '<br>\n'

df_entries = []

torch_tensor_entries = []
torch_tensor_entries_n = []

train_loader = []
test_loader = []

# add a subfolder for the losses
folderNameLosses = folderNameBranch + '/Losses/'
checkFolder(folderNameLosses)
print('\nfolderNameLosses : {:s}'.format(folderNameLosses))

lossesValues = folderNameLosses + "/lossesValues_" + branch + ".txt"
print("loss values file : %s\n" % lossesValues)
#wLoss = open(lossesValues, 'w')

tmp = df 
cols = df.columns.values
cols_entries = cols[6::2]
df_entries = tmp[cols_entries]
(_, Ncols) = df_entries.shape

# get nb of columns & rows for histos & remove over/underflow
(Nrows, Ncols) = df_entries.shape
print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branch))
df_entries = df_entries.iloc[:, 1:Ncols-1]
(Nrows, Ncols) = df_entries.shape
print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branch))
textHisto += 'nb of columns for histo {:s} after extraction : [{:3d}, {:3d}]<br>\n'.format(branch, Nrows, Ncols)

textHisto += '<br>\n'

# add a subfolder for the losses
folderNameLoader = folderNameBranch + '/TrainTestLOADER/'
checkFolder(folderNameLoader)
print('\nfolderNameLoader : {:s}'.format(folderNameLoader))

trainName = folderNameLoader + "multi_train_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
testName = folderNameLoader + "multi_test_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"

textHisto += 'creating train[test]_loader<br>\n'
textHisto += 'creating : {:s} OK.<br>\n'.format(trainName)
textHisto += 'creating : {:s} OK.<br>\n'.format(testName)
textHisto += '<br>\n'
# creating torch tensor from df_entries
torch_tensor_entries = torch.tensor(df_entries.values, device=device).float()
print('max df')
print(df_entries.values.max())
MAXMax = df_entries.values.max()
print('MAXMax : %e' % MAXMax)
if (MAXMax == 1.e38):
    print(colorText('ATTENTION, Kolossal PBM !!!', 'blue'))
# normalize the tensor
torch_tensor_entries_n = normalize(torch_tensor_entries, p=2.0)

train_size=int(percentageTrain*len(torch_tensor_entries)) # in general torch_tensor_entries = 200
test_size=len(torch_tensor_entries)-train_size
print('train size : %d' % (train_size))
print('test size  : %d' % (test_size))
textHisto += 'train size : {:d}<br>\n'.format(train_size)
textHisto += 'test size  : {:d}<br>\n'.format(test_size)
train_tmp, test_tmp = data.random_split(torch_tensor_entries_n,[train_size,test_size])

train_loader = data.DataLoader(train_tmp,batch_size=batch_size)
test_loader = data.DataLoader(test_tmp,batch_size=batch_size)

print('saving ... %s' % trainName)
torch.save(train_loader,trainName)
torch.save(test_loader,testName)
print('save OK.\n')
textHisto += 'save : {:s} OK.<br>\n'.format(trainName)
textHisto += 'save : {:s} OK.<br>\n'.format(testName)
textHisto += '<br>\n'

# load all data to device for gpu
loss_fn=torch.nn.MSELoss()

#define the network
textHisto += 'define the network (encoder/decoder)<br>\n'
encoder=Encoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
decoder=Decoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
textHisto += 'using <b>2</b> layers encoder/decoder<br>\n'
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
encoderName = folderNameLoader + "/mono_encoder_{:01d}_".format(nbLayer) + branch + "_{:03d}".format(nbFiles) + ".pth"
decoderName = folderNameLoader + "/mono_decoder_{:01d}_".format(nbLayer) + branch + "_{:03d}".format(nbFiles) + ".pth"
textHisto += 'encoderName : {:s}.<br>\n'.format(encoderName)
textHisto += 'decoderName : {:s}.<br>\n'.format(decoderName)
textHisto += '<br>\n'

# add a subfolder for the pictures
folderNamePict = folderNameBranch + '/Pictures/'
checkFolder(folderNamePict)
print('\nfolderNamePict : {:s}'.format(folderNamePict))

lossesPictureName = folderNamePict + '/loss_plots_' + branch + "_{:03d}".format(nbFiles) + '_V1.png'
textHisto += 'Calculating encoder/decoder<br>\n'
for epoch in range(nb_epochs):
    train_loss, encoded_out=train_epoch_den(encoder=encoder, decoder=decoder,device=device,
        dataloader=train_loader, loss_fn=loss_fn,optimizer=optim)
    test_loss, d_out, latent_out=test_epoch_den(encoder=encoder, decoder=decoder,device=device,
        dataloader=test_loader, loss_fn=loss_fn)
    L_out.append(d_out)
    LatentValues_Train.append(encoded_out.detach().numpy())
    LatentValues_Test.append(latent_out)
    history_da['train_loss'].append(train_loss)
    history_da['test_loss'].append(test_loss)

#r = (train_loss - test_loss) / (train_loss + test_loss)
#print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
#textHisto += 'epoch : {:03d} : train_loss = {:e} : test_loss = {:e}<br>\n'.format(epoch, train_loss, test_loss)
if ( saveEncoder == 1 ): # warning encoder & decoder are needed for next computations
    torch.save(encoder,encoderName)
    torch.save(decoder,decoderName)
    textHisto += 'save : {:s} OK.<br>\n'.format(encoderName)
    textHisto += 'save : {:s} OK.<br>\n'.format(decoderName)
textHisto += '<br>\n'

createLossPictures(branch, history_da, epoch+1, lossesPictureName)

wLoss = open(lossesValues, 'w')
wLoss.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))
wLoss.close()
fHisto = open(resumeHisto, 'w')  # web page
fHisto.write(textHisto)
fHisto.close()

time2 = time.time()

print('\n Recap ')
print('time for loop 1 : {:f}'.format(time2 - time1))
print('end')

