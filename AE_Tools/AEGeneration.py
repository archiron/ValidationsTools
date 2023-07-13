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

import sys, os
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("AEGeneration - arg. 0 :", sys.argv[0]) # name of the script
    print("AEGeneration - arg. 1 :", sys.argv[1]) # COMMON files path
    print("AEGeneration - arg. 2 :", sys.argv[2]) # FileName for paths
    print("AEGeneration - arg. 3 :", sys.argv[3]) # nb of datasets
    print("AEGeneration - arg. 4 :", sys.argv[4]) # dataset name
    print("AEGeneration - arg. 5 :", sys.argv[5]) # cpu/gpu option
    print("AEGeneration - arg. 6 :", sys.argv[6]) # timeFolder
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
    nbBranches = int(sys.argv[3])
    branch = sys.argv[4][1:]
    print('branch : {:s}'.format(branch))
    cg_pu = sys.argv[5]
    timeFolder = sys.argv[6]
else:
    print("rien")
    pathBase = ''
    cg_pu = ''

import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils import data
from torch.nn.functional import normalize

seed = 42
# Python RNG
random.seed(seed)
np.random.seed(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nAE Generation")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE
print('Lib path : {:s}'.format(pathChiLib))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import default as dfo
from default import *
from rootValues import NB_EVTS
from defaultStd import *
from autoEncoders import *
from controlFunctions import *
from graphicAutoEncoderFunctions import *
from sources import *

from DecisionBox import *
DB = DecisionBox()

def createAutoEncoderRef(nbFiles, nbBranches, device, lr, epsilon, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, latent_size, batch_size, nb_epochs, percentageTrain):
    Text = []
    Text.append(' <h1><center><b><font color=\'blue\'>AutoEncoder with {:d} ROOT files.</font></b></center></h1> <br>\n'.format(nbFiles))
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

arrayKSValues = []
rels = []

y_pred_n = []
y_pred_o = []

#nbBranches = len(branches) # [0:8]
print('there is {:03d} datasets'.format(nbBranches))

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathNb_evts, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathNb_files = pathCase + '/{:03d}'.format(nbFiles)
print('data_dir path : {:s}'.format(pathNb_files))
pathAE = pathNb_files + '/AE_RESULTS/'
print('data_res path : {:s}'.format(pathAE))

# get list of added ROOT files
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(pathDATA, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList)) + ' added ROOT files')
for item in rootFilesList:
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
for elem in sortedRels:
    print(elem)

# get list of text files
#pathKSFiles = pathNb_files
#print('KS path : %s' % pathKSFiles)
KSlistFiles = []
tmp = getListFiles(pathNb_files, 'txt')
for elem in tmp:
    if (elem[5:10] == '_diff'): # to keep only histo_differences_KScurves files
        KSlistFiles.append(elem)
print(KSlistFiles, len(KSlistFiles))
    
for item in KSlistFiles:
    print('file : %s' % item)
    aa = item.split('__')[0]
    fileName = pathNb_files + '/' + item
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
fileName = pathNb_files + "/branchesHistos_NewFiles.txt"
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

#timeFolder = time.strftime("%Y%m%d-%H%M%S") # only kept for tests

HL = [hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5, hidden_size_6, hidden_size_7]
useHL = [useHL1, useHL2, useHL3, useHL4, useHL5, useHL6, useHL7] # useHL1/HL2 always = 1.
#folderName = pathAE + createAEfolderName1(hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, useHL3, useHL4, latent_size)
folderName = pathAE + createAEfolderName(HL, useHL, latent_size)
checkFolder(folderName)
print('\nComplete folder name : {:s}'.format(folderName))

# export parameters of the layers
exportParameters = folderName + '/parameters.html'
print(exportParameters)
fParam = open(exportParameters, 'w')  # html page
for line in createAutoEncoderRef(nbFiles, nbBranches, device, lr, epsilon, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, latent_size, batch_size, nb_epochs, percentageTrain):
    fParam.write(line)
fParam.close()

print('{:s}\n'.format(branch))
df = []
fileName = pathNb_evts + "/histo_" + branch + '_{:03d}'.format(nbFiles) + ".txt"
if Path(fileName).exists():
    print('{:s} exist'.format(fileName))
    df = pd.read_csv(fileName)
else:
    print('{:s} does not exist'.format(fileName))
    exit()

# add a subfolder with the name of the histo and a folder with date/time
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
textHisto += "<title> Resume of ZEE_14 predictions"+ str(branch)+ " </title>\n"
textHisto += "</head>\n"

short_histo_name = reduceBranch(branch)
textHisto += ' <h1><center><b><font color=\'blue\'>{:s}</font></b></center></h1> <br>\n'.format(str(branch))
textHisto += '<b>folderName : </b>{:s}<br>\n'.format(folderNameBranch)
textHisto += '<br>\n'

df_entries = []
linOp = []

for line in Lines:
    rel,b = line.rstrip().split(',', 1)
    hName = b.rstrip().split(',', 1)[0]
    if ( str(hName) == str(branch)):
        linOp.append(line)

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

tmp = df
cols = df.columns.values
n_cols = len(cols)
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

if (useTrainLoader == 1):
    tmpPath = folderName + branch + '/' + TimeFolderRef + '/TrainTestLOADER/'
    trainName = tmpPath + "multi_train_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
    testName = tmpPath + "multi_test_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
    print('load %s.' % trainName)
    print('load %s.' % testName)
    if not os.path.isfile(trainName):
        print('%s does not exist' % trainName)
        textHisto += '{:s} does not exist. exiting.<br>\n'.format(trainName)
        exit()
    else:
        encoder = torch.load(trainName)
        train_loader = torch.load(trainName)
    if not os.path.isfile(testName):
        print('%s does not exist' % testName)
        textHisto += '{:s} does not exist. exiting.<br>\n'.format(testName)
        exit()
    else:
        textHisto += 'load {:s} and {:s}<br>\n'.format(trainName, testName)
        textHisto += '<br>\n'
        test_loader = torch.load(testName)
    print('load OK.')
else:
    textHisto += 'creating train[test]_loader<br>\n'
    textHisto += 'creating : {:s} OK.<br>\n'.format(trainName)
    textHisto += 'creating : {:s} OK.<br>\n'.format(testName)
    textHisto += '<br>\n'
    # creating torch tensor from df_entries/errors
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

loss_fn=torch.nn.MSELoss()

#define the network
textHisto += 'define the network (encoder/decoder)<br>\n'
HL = [hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5, hidden_size_6, hidden_size_7]
useHL = [useHL1, useHL2, useHL3, useHL4, useHL5, useHL6, useHL7] # useHL1/HL2 always = 1.
HL, useHL = extractLayerList(HL, useHL)

if useHL7 == 1:
    encoder=Encoder7(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5,hidden_size_6,hidden_size_7)
    decoder=Decoder7(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5,hidden_size_6,hidden_size_7)
    textHisto += 'using <b>4</b> layers encoder/decoder<br>\n'
    nbLayer = 4
elif useHL6 == 1:
    encoder=Encoder6(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5,hidden_size_6)
    decoder=Decoder6(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5,hidden_size_6)
    textHisto += 'using <b>4</b> layers encoder/decoder<br>\n'
    nbLayer = 4
elif useHL5 == 1:
    encoder=Encoder5(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5)
    decoder=Decoder5(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5)
    textHisto += 'using <b>3</b> layers encoder/decoder<br>\n'
    nbLayer = 3
elif useHL4 == 1:
    encoder=Encoder4(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
    decoder=Decoder4(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
    textHisto += 'using <b>4</b> layers encoder/decoder<br>\n'
    nbLayer = 4
elif useHL3 == 1:
    encoder=Encoder3(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3)
    decoder=Decoder3(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3)
    textHisto += 'using <b>3</b> layers encoder/decoder<br>\n'
    nbLayer = 3
else: # 2 layers
    encoder=Encoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
    decoder=Decoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
    textHisto += 'using <b>2</b> layers encoder/decoder<br>\n'
    nbLayer = 2

encoder.to(device)
decoder.to(device)

params_to_optimize = [
{'params': encoder.parameters()},
{'params': decoder.parameters()}
]

optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)
history_da = {'train_loss':[],'test_loss':[]}
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

lossesPictureName = folderNamePict + '/loss_plots_' + branch + "_{:03d}".format(nbFiles) + '.png'
if ( useEncoder == 1):
    textHisto += 'Using encoder/decoder<br>\n'
    if not os.path.isfile(encoderName):
        print('%s does not exist' % encoderName)
        textHisto += '{:s} does not exist. exiting.<br>\n'.format(encoderName)
        exit()
    else:
        encoder = torch.load(encoderName)
    if not os.path.isfile(decoderName):
        print('%s does not exist' % decoderName)
        textHisto += '{:s} does not exist. exiting.<br>\n'.format(decoderName)
        exit()
    else:
        decoder = torch.load(decoderName)
else:
    textHisto += 'Calculating encoder/decoder<br>\n'
    for epoch in range(nb_epochs):
        train_loss, encoded_out=train_epoch_den(encoder=encoder, decoder=decoder,device=device,
            dataloader=train_loader, loss_fn=loss_fn,optimizer=optim)
        test_loss, d_out, latent_out=test_epoch_den(encoder=encoder, decoder=decoder,device=device,
            dataloader=test_loader, loss_fn=loss_fn)
        L_out.append(d_out)
        LatentValues_Train.append(encoded_out.detach().numpy())
        LatentValues_Test.append(latent_out)
        #r = (train_loss - test_loss) / (train_loss + test_loss)
        #print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
        history_da['train_loss'].append(train_loss)
        history_da['test_loss'].append(test_loss)

        '''bo1 = train_loss < epsilon #deja commenté
        bo2 = test_loss < epsilon#deja commenté
        if (bo1 and bo2):#deja commenté
            break'''#deja commenté

    r = (train_loss - test_loss) / (train_loss + test_loss)
    print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
    textHisto += 'epoch : {:03d} : train_loss = {:e} : test_loss = {:e}<br>\n'.format(epoch, train_loss, test_loss)
    #print('epoch : %03d : tr_lo = %e : te_lo = %e' % (epoch, train_loss, test_loss))
    if ( saveEncoder == 1 ): # warning encoder & decoder are needed for next computations
        torch.save(encoder,encoderName)
        torch.save(decoder,decoderName)
        textHisto += 'save : {:s} OK.<br>\n'.format(encoderName)
        textHisto += 'save : {:s} OK.<br>\n'.format(decoderName)
    textHisto += '<br>\n'

    #print('write HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e'
    #            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))
    #createLossPictures(branch, history_da, nb_epochs, lossesPictureName)
    createLossPictures(branch, history_da, epoch+1, lossesPictureName)

    x_Train = []
    y_Train = []
    x_Test = []
    y_Test = []
    title='Train/Test latent picture in 2 dim'
    pictureName = folderNamePict + '/traintestLatentPicture_' + branch + '.png'
    #pictureName2 = folderNamePict + '/traintestLatentPicture2_' + branch + '.png'
    for ind in range(0, len(LatentValues_Train)):
        x_Train.append(LatentValues_Train[ind][0])
        y_Train.append(LatentValues_Train[ind][1])
    
    for ind in range(0, len(LatentValues_Test)):
        #print('Test ', ind, LatentValues_Test[ind][0])
        x_Test.append(LatentValues_Test[ind][0].numpy()[0])
        y_Test.append(LatentValues_Test[ind][0].numpy()[1])
    print('createLatentPictureTrainTest call')
    createLatentPictureTrainTest(x_Train,y_Train,x_Test,y_Test, pictureName, title)
    #createLatentPictureTrainTest(x_Test,y_Test,x_Test,y_Test, pictureName2, title)

    nb_history_da = len(history_da['train_loss'])
    t1 = np.asarray(history_da['train_loss'])
    t2 = np.asarray(history_da['test_loss'])
    rr = 1.
    ss = 1.
    for kk in range(nb_history_da-10,nb_history_da):
        rr *= t1[kk]/t2[kk]
        ss *= t2[kk]/t1[kk]
    print('coefficient losses : {:1.4e}'.format(rr) + ' - {:1.4e}'.format(ss))

# Ready for prediction
print('using %s\n' % encoderName)
textHisto += 'Using {:s} for prediction<br>\n'.format(encoderName)
textHisto += '<br>\n'

predLossesValues = folderNameBranch + "/predLossesValues_" + branch + ".txt"
print("loss values file : %s" % predLossesValues)
wPred = open(predLossesValues, 'w')

# export the y_pred_new values
predValues = folderNameBranch + "/predValues_" + branch + ".txt"
print("values file : %s" % predValues)
wPredVal = open(predValues, 'w')

lossesVal = []
latentVal = []
LinesPred = []
for elem in linOp:
    print('linOp : ', elem)
    rel, hName,line = elem.rstrip().split(',', 2)
    #print(rel,hName,line)
    new = line.rstrip().split(',')
    new = np.asarray(new).astype(float)
    #print(new)
    df_new = pd.DataFrame(new).T # otherwise, one column with 50 rows instead of 1 line with 50 columns

    # creating torch tensor from df_entries/errors
    torch_tensor_new = torch.tensor(df_new.values, device=device).float()

    # normalize the tensor
    '''if '_pfx' in branch:
        print('pfx')
        torch_tensor_entries_n = torch_tensor_new
    else:
        torch_tensor_entries_n = normalize(torch_tensor_new, p=2.0)'''
    torch_tensor_entries_n = normalize(torch_tensor_new, p=2.0)
    test_loader_n = data.DataLoader(torch_tensor_entries_n)

    encoder = torch.load(encoderName)
    decoder = torch.load(decoderName)
    loss_fn = torch.nn.MSELoss()

    params_to_optimize=[
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
    ]

    optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)

    # Forward pass: Compute predicted y by passing x to the model
    new_loss, y_pred_new, latent_out = test_epoch_den(encoder=encoder,
            decoder=decoder,device=device,
            dataloader=test_loader_n,
            loss_fn=loss_fn)

    # Compute and print loss
    #print('new loss value : %e for %s' % (new_loss, rel))
    wPred.write('%e, %s\n' % (new_loss, rel))
    #textHisto += 'new loss value : {:e} for {:s}<br>\n'.format(new_loss, rel)
    lossesVal.append([rel,new_loss.item()])
    latentVal.append(latent_out[0].numpy())

    pictureName = folderNamePict + '/predicted_new_curves_' + branch + '_' + rel[6:] + '_multi.png'
    ### WARNING rel is the same for all comparisons !!!
    creatPredPictLinLog(branch, Ncols, torch_tensor_entries_n, y_pred_new, new_loss, rel[6:], pictureName)

    # write values into the predValues file (# export the y_pred_new values)
    text2write = rel + ',' + branch
    for val in y_pred_new.numpy():
        N=len(val)
        for nn in range(0,N):
            text2write += ',' + str(val[nn])
    LinesPred.append(text2write)
    text2write += '\n'
    wPredVal.write(text2write)

labels = []
val = []
x = []
y = []

sortedLossesVal = sorted(lossesVal, key = lambda x: x[0])
for elem in sortedLossesVal:
    labels.append(elem[0][6:])
    val.append(elem[1])
mn = np.mean(val)
# a priori ne sert pas bcp.
textHisto += '<table border="0" bordercolor=red cellpadding="2">' + '\n'
textHisto += "<tr><td>\n"

textHisto += '<table border="1" bordercolor=green cellpadding="2">' + '\n'
textHisto += "<tr><td>\n"
textHisto += 'Release</td><td>new loss value</td>'
for elem in sortedLossesVal:
    r = np.abs((elem[1]-mn)/(elem[1]+mn))
    print('new loss value for {:20s} : {:e}'.format(elem[0], elem[1]))
    textHisto += "<tr><td>"
    textHisto += '{0:20s}'.format(elem[0][6:])
    textHisto += "</td><td>"
    textHisto += '{:e}'.format(elem[1])
    textHisto += "</td></tr>\n"
textHisto += "\n</table>\n"
textHisto += "</td>"
pictureName = folderNamePict + '/comparison_loss_values_' + branch + '.png'
title = r"$\bf{" + branch + "}$" + ' : Losses vs releases.'
createCompLossesPicture(labels,val, pictureName, title)

title='Latent ReleasesVsTrain comparison in 2 dim'
pictureName = folderNamePict + '/LatentReleasesVsTrainPicture_' + branch + '.png' 
for ind, text in enumerate(labels):
    x.append(latentVal[ind][0])
    y.append(latentVal[ind][1])
createLatentPicture(labels,x,y, pictureName, title)
pictureName = folderNamePict + '/compLatentReleasesVsTrainPicture_' + branch + '.png'
createCompLatentPictureTrainTest(labels, x_Train,y_Train,x,y, pictureName, title)

wPred.close()
wPredVal.close()
print('end of %s' % branch)

KSLoss = []
for elem in sortedArrayKSValues:
    if elem[0] == branch:
        print(elem,elem[1], elem[2])
        KSLoss.append([elem[1], elem[2]])
sortedKSLoss = sorted(KSLoss, key = lambda x: x[0]) # gives an array with releases sorted
if (len(sortedKSLoss) > 0):
    print('sortedKSLoss OK')
else:
    print('sortedKSLoss KO')
    exit() # continue

print('sortedLossesVal')
print(sortedLossesVal)
print('sortedKSLoss')
print(sortedKSLoss)
print('sortedRels')
print(sortedRels)

labels = []
Val1 = []
Val2 = []
print('arr ok')
for j in range(0, len(sortedRels)):
    print(j, sortedRels[j], [sortedLossesVal[j][1], sortedKSLoss[j][1]])
    Val1.append(sortedLossesVal[j][1])
    Val2.append(sortedKSLoss[j][1])
    labels.append(sortedRels[j][1])

fileName = folderNamePict + 'compLossesValuesVsKS_' + branch + '.png'
title = r"$\bf{" + branch + "}$" + ' : Losses, KS pValues vs releases.'
createCompLossesPicture2Axis(labels, Val1, Val2, fileName, title)

# extract the values of originals curves from Lines
LinesOrig = []
for l in Lines:
    l_split = l.split(',')[1]
    #print(l_split)
    if ( l_split == branch ):
        LinesOrig.append(l)
#for l in range(0,len(LinesOrig)):
#    print('{:02d}/{:02d} : {:s}'.format(l,12,LinesOrig[l]))

print('len LinesPred : {}'.format(len(LinesPred)))
print('len LinesOrig : {}'.format(len(LinesOrig)))

labels = sortedRels
labels1 = []
val1 = []
for ll in labels:
    predKSValues = pathNb_files + "/histo_differences_KScurve_" + ll[1] + "__{:03d}".format(nbFiles) + ".txt"
    print("values file : %s" % predKSValues)
    wPredKSVal = open(predKSValues, 'r')
    LinesKSVal = wPredKSVal.readlines()
    #for l in range(0,len(LinesKSVal)):
    #    print('{:02d}/{:02d} : {:s}'.format(l,259,LinesKSVal[l]))
    wPredKSVal.close()
    labels1.append(ll[1])
    for l in range(0, len(LinesKSVal)):
        l_pred = LinesKSVal[l][:-1].split(' : ')
        if ( l_pred[0] == branch ):
            val1.append(float(l_pred[1]))

labels2 = []
val2 = []
tmp2 = []
# create the curves for each release
for l in range(0, len(LinesPred)):
    l_pred = LinesPred[l][:-1].split(',')
    #relise_p = l_pred[0]
    l_orig = LinesOrig[l][:-1].split(',')
    relise_o = l_pred[0]
    #print(relise_p, relise_o) # , rels[l]
    l_pred = np.asarray(l_pred[2:])
    l_pred = l_pred.astype(np.float64)
    l_orig = np.asarray(l_orig[2:])
    l_orig = l_orig.astype(np.float64)
    diff_max, _, l_diff = DB.diffMAXKS3(l_pred, l_orig)
    tmp2.append([relise_o[6:], diff_max])
    #print(l_diff)
#print(tmp2)
for l in labels1:
    for m in tmp2:
        if (l == m[0]):
            labels2.append(l)
            val2.append(m[1])

if (len(val1) > 0):
    pictureName = folderNamePict + 'comparison_KSvsAE_' + branch + '_{:03d}'.format(nbFiles) +'.png' # 
    print(pictureName)
    title = r"$\bf{" + branch + "}$" + ' : KS diff. max., AE losses vs releases.'
    createCompKSvsAEPicture(labels2, val1, val2, pictureName, title)
    pictureName = folderNamePict + 'comparison_KSvsAE_2Axis_' + branch + '_{:03d}'.format(nbFiles) +'.png' # give a name whith data_CMS/cms/chiron/Validations/1000/DEV_01//004/AE_RESULTS/
    createCompKSvsAEPicture2Axis(labels2, val1, val2, pictureName, title) # same as above with 2 vert. axis instead of one.

pictureName = pictureName.replace('/data_CMS/cms/chiron/Validations', 'https://llrvalidation.in2p3.fr')
histoPath = 'https://llrvalidation.in2p3.fr/' + '/' + resumeHisto
textHisto += '<td><a href=\"' + pictureName + '\"><img width=\"450\" height=\"250\" border=\"0\" align=\"middle\" src=\"' + pictureName + '\"></a></td>'
lossPath = lossesPictureName.replace('/data_CMS/cms/chiron/Validations', 'https://llrvalidation.in2p3.fr')
textHisto += '<br>'
textHisto += '<td><a href=\"' + lossPath + '\"><img width=\"450\" height=\"250\" border=\"0\" align=\"middle\" src=\"' + lossPath + '\"></a></td>'
textHisto += "</tr>\n"

textHisto += "<tr><td>\n"
textHisto += '<table border="1" bordercolor=green cellpadding="2">' + '\n'
textHisto += "<tr>\n"
textHisto += '<td>Release</td><td>new loss value</td>'
for elem in sortedLossesVal:
    r = np.abs((elem[1]-mn)/(elem[1]+mn))
    #print('new loss value for {:20s} : {:e}'.format(elem[0], elem[1]))
    textHisto += "<tr><td>"
    textHisto += '{0:20s}'.format(elem[0][6:])
    textHisto += "</td><td>"
    #textHisto += '{:e}'.format(elem[1])
    pictureName2 = folderNamePict + '/predicted_new_curves_' + branch + '_' + elem[0][6:] + '_multi.png' # folderNamePict + 
    pictureName2 = pictureName2.replace('/data_CMS/cms/chiron/Validations', 'https://llrvalidation.in2p3.fr')
    #textHisto += '{:s}'.format(pictureName2)
    textHisto += '<a href=\"' + pictureName2 + '\"><img width=\"250\" height=\"125\" border=\"0\" align=\"middle\" src=\"' + pictureName2 + '\"></a>'
    textHisto += "</td></tr>\n"
textHisto += "\n</table>\n"
textHisto += "</td></tr>"

textHisto += '</table>'

wLoss = open(lossesValues, 'w')
wLoss.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))
wLoss.close()

fHisto = open(resumeHisto, 'w')  # web page
fHisto.write(textHisto)
fHisto.close()
print('end')

