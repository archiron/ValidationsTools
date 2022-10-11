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
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
## WARNING pbm with torch
import torch
from torch.utils import data
from torch.nn.functional import normalize

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nAE Generation")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from rootValues import NB_EVTS
from defaultStd import *
from autoEncoders import *
from controlFunctions import *
from graphicAutoEncoderFunctions import *

from DecisionBox import *
DB = DecisionBox()

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

df = []
arrayKSValues = []
rels = []

#H_da = []
#n_loss = []
#o_loss = []
y_pred_n = []
y_pred_o = []

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

#histoKeysNames = getKeysName(tp_1, source) # unused
#print(len(histoKeysNames))
    
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
data_res = data_dir + '/AE_RESULTS/'
print('data_res path : {:s}'.format(data_res))

for branch in branches: # [0:8]
    fileName = resultPath + "/histo_" + branch + '_{:03d}'.format(nbFiles) + ".txt"
    if Path(fileName).exists():
        print('{:s} exist'.format(fileName))
        df.append(pd.read_csv(fileName))
    else:
        print('{:s} does not exist'.format(fileName))

# get list of added ROOT files
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList)) + ' added ROOT files')
for item in rootFilesList:
    #print('%s' % item)
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
    aa = item[26:-9]
    fileName = pathKSFiles + '/' + item
    file1 = open(fileName, 'r')
    bb = file1.readlines()
    for elem in bb:
        tmp = []
        cc = elem.split(' : ')
        tmp = [cc[0], aa, float(cc[1][:-1])]
        arrayKSValues.append(tmp)
sortedArrayKSValues = sorted(arrayKSValues, key = lambda x: x[0]) # gives an array with releases sorted
#for elem in sortedArrayKSValues:
#    print(elem)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device : {:s}'.format(str(device)))

t = datetime.datetime.today()
timeFolder = time.strftime("%Y%m%d-%H%M%S")

folderName = data_res + createAEfolderName(hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, useHL3, useHL4, latent_size) # , timeFolder, nbFiles, branches[i]
checkFolder(folderName)
print('\nComplete folder name : {:s}'.format(folderName))

# export parameters of the layers
exportParameters = folderName + '/parameters.html'
print(exportParameters)
fParam = open(exportParameters, 'w')  # html page
for line in createAutoEncoderRef(nbFiles, nbBranches, device, lr, epsilon, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, latent_size, batch_size, nb_epochs, percentageTrain):
    fParam.write(line)
fParam.close()

loopMaxValue = nbBranches # nbBranches
for i in range(0, loopMaxValue):

    # add a subfolder with the name of the histo and a folder with date/time
    folderNameBranch = folderName + branches[i] + '/' + timeFolder
    checkFolder(folderNameBranch)
    print('\n===== folderNameBranch : {:s} ====='.format(folderNameBranch))

    resumeHisto = folderNameBranch + '/histo_' + '{:s}'.format(str(branches[i]))
    resumeHisto += '.html'
    print(resumeHisto)
    fHisto = open(resumeHisto, 'w')  # web page
    fHisto.write("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n")
    fHisto.write("<html>\n")
    fHisto.write("<head>\n")
    fHisto.write("<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n")
    fHisto.write("<title> Resume of ZEE_14 predictions"+ str(branches[i])+ " </title>\n")  # 
    fHisto.write("</head>\n")

    short_histo_name = reduceBranch(branches[i])
    fHisto.write(' <h1><center><b><font color=\'blue\'>{:s}</font></b></center></h1> <br>\n'.format(str(branches[i])))
    fHisto.write('<b>folderName : </b>{:s}<br>\n'.format(folderNameBranch))
    fHisto.write('<br>\n')
    
    df_entries = []
    df_errors = []
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

    lossesValues = folderNameLosses + "/lossesValues_" + branches[i] + ".txt"
    print("loss values file : %s\n" % lossesValues)
    wLoss = open(lossesValues, 'w')

    tmp = df[i]
    cols = df[i].columns.values
    n_cols = len(cols)
    #print('nb of columns for histo {:s} : {:d}'.format(branches[i], n_cols))
    #fHisto.write('nb of columns for histo {:s} : {:d}<br>\n'.format(branches[i], n_cols))
    cols_entries = cols[6::2]
    cols_errors = cols[7::2]
    df_entries = tmp[cols_entries]
    (_, Ncols) = df_entries.shape

    # get nb of columns & rows for histos & remove over/underflow
    (Nrows, Ncols) = df_entries.shape
    print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branches[i]))
    df_entries = df_entries.iloc[:, 1:Ncols-1]
    (Nrows, Ncols) = df_entries.shape
    print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branches[i]))
    fHisto.write('nb of columns for histo {:s} after extraction : [{:3d}, {:3d}]<br>\n'.format(branches[i], Nrows, Ncols))

    fHisto.write('<br>\n')

    # add a subfolder for the losses
    folderNameLoader = folderNameBranch + '/TrainTestLOADER/'
    checkFolder(folderNameLoader)
    print('\nfolderNameLoader : {:s}'.format(folderNameLoader))

    trainName = folderNameLoader + "multi_train_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    testName = folderNameLoader + "multi_test_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"

    if (useTrainLoader == 1):
        tmpPath = folderName + branches[i] + '/' + TimeFolderRef + '/TrainTestLOADER/'
        trainName = tmpPath + "multi_train_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
        testName = tmpPath + "multi_test_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
        print('load %s.' % trainName)
        print('load %s.' % testName)
        if not os.path.isfile(trainName):
            print('%s does not exist' % trainName)
            fHisto.write('{:s} does not exist. exiting.<br>\n'.format(trainName))
            exit()
        else:
            encoder = torch.load(trainName)
            train_loader = torch.load(trainName)
        if not os.path.isfile(testName):
            print('%s does not exist' % testName)
            fHisto.write('{:s} does not exist. exiting.<br>\n'.format(testName))
            exit()
        else:
            fHisto.write('load {:s} and {:s}<br>\n'.format(trainName, testName))
            fHisto.write('<br>\n')
            test_loader = torch.load(testName)
        print('load OK.')
    else:
        fHisto.write('creating train[test]_loader<br>\n')
        fHisto.write('creating : {:s} OK.<br>\n'.format(trainName))
        fHisto.write('creating : {:s} OK.<br>\n'.format(testName))
        fHisto.write('<br>\n')
        # creating torch tensor from df_entries/errors
        torch_tensor_entries = torch.tensor(df_entries.values)
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
        print('%d : train size : %d' % (i,train_size))
        print('%d : test size  : %d' % (i,test_size))
        fHisto.write('train size : {:d}<br>\n'.format(train_size))
        fHisto.write('test size  : {:d}<br>\n'.format(test_size))
        train_tmp, test_tmp = data.random_split(torch_tensor_entries_n,[train_size,test_size])
    
        train_loader = data.DataLoader(train_tmp,batch_size=batch_size)
        test_loader = data.DataLoader(test_tmp,batch_size=batch_size)
        
        print('saving ... %s' % trainName)
        torch.save(train_loader,trainName)
        torch.save(test_loader,testName)
        print('save OK.\n')
        fHisto.write('save : {:s} OK.<br>\n'.format(trainName))
        fHisto.write('save : {:s} OK.<br>\n'.format(testName))
        fHisto.write('<br>\n')

    loss_fn=torch.nn.MSELoss()

    #define the network
    fHisto.write('define the network (encoder/decoder)<br>\n')
    if useHL4 == 1:
        encoder=Encoder4(latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
        decoder=Decoder4(latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
        fHisto.write('using <b>4</b> layers encoder/decoder<br>\n')
        nbLayer = 4
    elif useHL3 == 1:
        encoder=Encoder3(latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3)
        decoder=Decoder3(latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3)
        fHisto.write('using <b>3</b> layers encoder/decoder<br>\n')
        nbLayer = 3
    else: # 2 layers
        encoder=Encoder2(latent_size,Ncols,hidden_size_1,hidden_size_2)
        decoder=Decoder2(latent_size,Ncols,hidden_size_1,hidden_size_2)
        fHisto.write('using <b>2</b> layers encoder/decoder<br>\n')
        nbLayer = 2

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
    fHisto.write('encoderName : {:s}.<br>\n'.format(encoderName))
    fHisto.write('decoderName : {:s}.<br>\n'.format(decoderName))
    fHisto.write('<br>\n')

    # add a subfolder for the pictures
    folderNamePict = folderNameBranch + '/Pictures/'
    checkFolder(folderNamePict)
    print('\nfolderNamePict : {:s}'.format(folderNamePict))

    lossesPictureName = folderNamePict + '/loss_plots_' + branches[i] + "_{:03d}".format(nbFiles) + '.png'
    if ( useEncoder == 1):
        fHisto.write('Using encoder/decoder<br>\n')
        if not os.path.isfile(encoderName):
            print('%s does not exist' % encoderName)
            fHisto.write('{:s} does not exist. exiting.<br>\n'.format(encoderName))
            exit()
        else:
            encoder = torch.load(encoderName)
        if not os.path.isfile(decoderName):
            print('%s does not exist' % decoderName)
            fHisto.write('{:s} does not exist. exiting.<br>\n'.format(decoderName))
            exit()
        else:
            decoder = torch.load(decoderName)
    else:
        fHisto.write('Calculating encoder/decoder<br>\n')
        for epoch in range(nb_epochs):
            train_loss, encoded_out=train_epoch_den(encoder=encoder, decoder=decoder,device=device,
                dataloader=train_loader, loss_fn=loss_fn,optimizer=optim)
            #print('epoch : ', epoch, encoded_out.detach().numpy())
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
        fHisto.write('epoch : {:03d} : train_loss = {:e} : test_loss = {:e}<br>\n'.format(epoch, train_loss, test_loss))
        #print('epoch : %03d : tr_lo = %e : te_lo = %e' % (epoch, train_loss, test_loss))
        if ( saveEncoder == 1 ): # warning encoder & decoder are needed for next computations
            torch.save(encoder,encoderName)
            torch.save(decoder,decoderName)
            fHisto.write('save : {:s} OK.<br>\n'.format(encoderName))
            fHisto.write('save : {:s} OK.<br>\n'.format(decoderName))
        fHisto.write('<br>\n')

        #print('write HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e'
        #            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))
        wLoss.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
                    % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))

        #createLossPictures(branches[i], history_da, nb_epochs, lossesPictureName)
        createLossPictures(branches[i], history_da, epoch+1, lossesPictureName)

        labels_Train = []
        labels_Test = []
        x_Train = []
        y_Train = []
        x_Test = []
        y_Test = []
        title='Train/Test latent picture in 2 dim'
        pictureName = folderNamePict + '/traintestLatentPicture_' + branches[i] + '.png'
        #pictureName2 = folderNamePict + '/traintestLatentPicture2_' + branches[i] + '.png'
        for ind in range(0, len(LatentValues_Train)):
            #print('Train ', ind, LatentValues_Train[ind])
            x_Train.append(LatentValues_Train[ind][0])
            y_Train.append(LatentValues_Train[ind][1])
            labels_Train.append(i)
        
        for ind in range(0, len(LatentValues_Test)):
            #print('Test ', ind, LatentValues_Test[ind][0])
            x_Test.append(LatentValues_Test[ind][0].numpy()[0])
            y_Test.append(LatentValues_Test[ind][0].numpy()[1])
            labels_Test.append(i)
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
    fHisto.write('Using {:s} for prediction<br>\n'.format(encoderName))
    fHisto.write('<br>\n')

    predLossesValues = folderNameBranch + "/predLossesValues_" + branches[i] + ".txt"
    print("loss values file : %s" % predLossesValues)
    wPred = open(predLossesValues, 'w')

    # export the y_pred_new values
    predValues = folderNameBranch + "/predValues_" + branches[i] + ".txt"
    print("values file : %s" % predValues)
    wPredVal = open(predValues, 'w')

    lossesVal = []
    latentVal = []
    LinesPred = []
    for elem in linOp:
        #print(elem)
        rel, hName,line = elem.rstrip().split(',', 2)
        #print(rel,hName,line)
        new = line.rstrip().split(',')
        new = np.asarray(new).astype(float)

        df_new = pd.DataFrame(new).T # otherwise, one column with 50 rows instead of 1 line with 50 columns

        # creating torch tensor from df_entries/errors
        torch_tensor_new = torch.tensor(df_new.values)

        # normalize the tensor
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
        #fHisto.write('new loss value : {:e} for {:s}<br>\n'.format(new_loss, rel))
        lossesVal.append([rel,new_loss.item()])
        latentVal.append(latent_out[0].numpy())
        #print(torch_tensor_entries_n)
        #print(y_pred_new)

        pictureName = folderNamePict + '/predicted_new_curves_' + branches[i] + '_' + rel[6:] + '_multi.png'
        ### WARNING rel is the same for all comparisons !!!
        creatPredPictLinLog(branches[i], Ncols, torch_tensor_entries_n, y_pred_new, new_loss, rel[6:], pictureName)

        # write values into the predValues file (# export the y_pred_new values)
        text2write = rel + ',' + branches[i]
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
    #print(val, mn)
    # a priori ne sert pas bcp.
    fHisto.write('<table border="0" bordercolor=red cellpadding="2">' + '\n')
    fHisto.write("<tr><td>\n")

    fHisto.write('<table border="1" bordercolor=green cellpadding="2">' + '\n')
    fHisto.write("<tr><td>\n")
    fHisto.write('new loss value</td><td>Release</td>')
    for elem in sortedLossesVal:
        r = np.abs((elem[1]-mn)/(elem[1]+mn))
        #print('{:^{width}} : '.format(elem[0], width=20) + '{:1.4e}'.format(r))
        #print('{:20s} : '.format(elem[0]) + '{:1.4e}'.format(r))
        print('new loss value for {:20s} : {:e}'.format(elem[0], elem[1]))
        fHisto.write("<tr><td>")
        fHisto.write('{0:20s}'.format(elem[0][6:]))
        fHisto.write("</td><td>")
        fHisto.write('{:e}'.format(elem[1]))
        fHisto.write("</td></tr>\n")
    fHisto.write("\n</table>\n")
    fHisto.write("</td>")
    pictureName = folderNamePict + '/comparison_loss_values_' + branches[i] + '.png'
    title = r"$\bf{" + branches[i] + "}$" + ' : Comparison of the losses values as function of releases.'
    createCompLossesPicture(labels,val, pictureName, title)

    title='Latent ReleasesVsTrain comparison in 2 dim'
    pictureName = folderNamePict + '/LatentReleasesVsTrainPicture_' + branches[i] + '.png' # '.svg'
    for ind, text in enumerate(labels):
        #print(text, latentVal[ind])
        x.append(latentVal[ind][0])
        y.append(latentVal[ind][1])
    createLatentPicture(labels,x,y, pictureName, title)
    pictureName = folderNamePict + '/compLatentReleasesVsTrainPicture_' + branches[i] + '.png'
    createCompLatentPictureTrainTest(labels, x_Train,y_Train,x,y, pictureName, title)
    
    wPred.close()
    wLoss.close()
    wPredVal.close()
    print('end of %s' % branches[i])
    
    KSLoss = []
    for elem in sortedArrayKSValues:
        if elem[0] == branches[i]:
            #print(elem,elem[1], elem[2])
            KSLoss.append([elem[1], elem[2]])
    sortedKSLoss = sorted(KSLoss, key = lambda x: x[0]) # gives an array with releases sorted
    #print('sortedKSLoss : ', sortedKSLoss)
    if (len(sortedKSLoss) > 0):
        print('sortedKSLoss OK')
    else:
        print('sortedKSLoss KO')
        continue

    labels = []
    Val1 = []
    Val2 = []
    print('arr ok')
    for j in range(0, len(sortedRels)):
        #print(j, sortedRels[j], [sortedArrLoss[j][1], sortedKSLoss[j][1]])
        Val1.append(sortedLossesVal[j][1])
        Val2.append(sortedKSLoss[j][1])
        labels.append(sortedRels[j][1])
    '''#print(Val1)
    #print(Val2)
    #print(labels)'''

    #print(branches[i])
    fileName = folderNamePict + 'compLossesValuesVsKS_' + branches[i] + '.png'
    title = r"$\bf{" + branches[i] + "}$" + ' : Losses values vs KS values as function of releases.'
    createCompLossesPicture2Axis(labels, Val1, Val2, fileName, title)
    
    # extract the values of originals curves from Lines
    LinesOrig = []
    for l in Lines:
        l_split = l.split(',')[1]
        #print(l_split)
        if ( l_split == branches[i] ):
            LinesOrig.append(l)
    #for l in range(0,len(LinesOrig)):
    #    print('{:02d}/{:02d} : {:s}'.format(l,12,LinesOrig[l]))

    print('len LinesPred : {}'.format(len(LinesPred)))
    print('len LinesOrig : {}'.format(len(LinesOrig)))

    labels = sortedRels
    labels1 = []
    val1 = []
    for ll in labels:
        predKSValues = data_dir + "/histo_differences_KScurve_" + ll[1] + "__{:03d}".format(nbFiles) + "_v2.txt"
        print("values file : %s" % predKSValues)
        wPredKSVal = open(predKSValues, 'r')
        LinesKSVal = wPredKSVal.readlines()
        #for l in range(0,len(LinesKSVal)):
        #    print('{:02d}/{:02d} : {:s}'.format(l,259,LinesKSVal[l]))
        wPredKSVal.close()
        labels1.append(ll[1])
        for l in range(0, len(LinesKSVal)):
            l_pred = LinesKSVal[l][:-1].split(' : ')
            if ( l_pred[0] == branches[i] ):
                val1.append(float(l_pred[1]))
    #print('labels1 ', labels1)
    #print(val1)
    #print('')

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
    #print('labels2 ', labels2)
    #print(val2)

    if (len(val1) > 0):
        pictureName = folderNamePict + 'comparison_KSvsAE_' + branches[i] + '_{:03d}'.format(nbFiles) +'.png' # 
        #print(pictureName)
        title = r"$\bf{" + branches[i] + "}$" + ' : Comparison of KS vs AE values as function of releases.'
        #print(title)
        createCompKSvsAEPicture(labels2, val1, val2, pictureName, title)
        pictureName = folderNamePict + 'comparison_KSvsAE_2Axis_' + branches[i] + '_{:03d}'.format(nbFiles) +'.png' # 
        createCompKSvsAEPicture2Axis(labels2, val1, val2, pictureName, title)

    #pictureName = os.getcwd() + '/' + pictureName
    pictureName = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + '/' + pictureName
    histoPath = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + '/' + resumeHisto
    fHisto.write('<td><a href=\"' + pictureName + '\"><img width=\"450\" height=\"250\" border=\"0\" align=\"middle\" src=\"' + pictureName + '\"></a></td>')
    lossPath = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + '/' + lossesPictureName
    fHisto.write('<td><a href=\"' + lossPath + '\"><img width=\"450\" height=\"250\" border=\"0\" align=\"middle\" src=\"' + lossPath + '\"></a></td>')
    fHisto.write("</tr>\n")

    fHisto.write("<tr><td>\n")
    fHisto.write('<table border="1" bordercolor=green cellpadding="2">' + '\n')
    fHisto.write("<tr><td>\n")
    fHisto.write('new loss value</td><td>Release</td>')
    for elem in sortedLossesVal:
        r = np.abs((elem[1]-mn)/(elem[1]+mn))
        #print('new loss value for {:20s} : {:e}'.format(elem[0], elem[1]))
        fHisto.write("<tr><td>")
        fHisto.write('{0:20s}'.format(elem[0][6:]))
        fHisto.write("</td><td>")
        #fHisto.write('{:e}'.format(elem[1]))
        pictureName2 = folderNamePict + '/predicted_new_curves_' + branches[i] + '_' + elem[0][6:] + '_multi.png'
        pictureName2 = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + '/' + pictureName2
        #fHisto.write('{:s}'.format(pictureName2))
        fHisto.write('<a href=\"' + pictureName2 + '\"><img width=\"250\" height=\"125\" border=\"0\" align=\"middle\" src=\"' + pictureName2 + '\"></a>')
        fHisto.write("</td></tr>\n")
    fHisto.write("\n</table>\n")
    fHisto.write("</td></tr>")

    fHisto.write('</table>')

fHisto.close()
print('end')

