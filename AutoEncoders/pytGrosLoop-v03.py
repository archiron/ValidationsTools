#!/usr/bin/env python
# coding: utf-8

import datetime
import os
from turtle import title

import numpy as np
import torch
#from torch import nn
from torch.utils import data
from torch.nn.functional import normalize
import pandas as pd

from autoEncoders import *
from controlFunctions import *
from graph import *
from defaultStd import *
from default import *

def getKeysName(t_p, branchPath):
    b = []
    key = ''
    tmp = []
    source = open(branchPath, "r")
    for line in source:
        if line in ['\n', '\r\n']: # blank line
            if ( (len(key) != 0) and (len(tmp) != 0) ):
                b.append([key, tmp])
                key = ''
                tmp = []
        else: # line not empty
            if t_p in line:
                aaa = line.split(' ')
                bbb = []
                for elem in aaa:
                    if elem != '':
                        bbb.append(elem)
                line = bbb[0].split('/')[1].replace(t_p, '')
                name = line.split(' ')[0]
                tmp.append([name, bbb[3]]) 
            else:
                key = line
    source.close()
    return b

def testExtension(histoName, histoPrevious):
    after = "" # $histoName
    common = histoName
    if ( '_' not in histoName ): # no _ in histo name
        before = histoName
        common = histoName
    else:
        afters = histoName.split('_')
        before = ''
        nMax = len(afters)
        #print('nMax : %d, histoprevious : %s'%(nMax,histoPrevious))

        if ( afters[nMax - 1] == "endcaps" ):
            after = "endcaps"
            for i in range(0, nMax-1):
                before += afters[i] + "_"
                before = before[:-1]
        elif ( afters[nMax - 1] == "barrel" ):
            after = "barrel"
            for i in range(0, nMax-1):
                before += afters[i] + "_"
                before = before[:-1]
        else:
            if ( histoPrevious == '' ):
                before = histoName
                after = ''
                common = histoName
            else:
                avant = '' # afters[0]
                after = ''
                for i in range(0, nMax-1):
                    avant = avant + "_" + afters[i]
                    avant = avant[1:]
                    if ( avant == histoPrevious ):
                        #print('yep')
                        before = avant
                        common = histoPrevious
                        break
                for j in range(nMax - i, nMax):
                    after += "_" + afters[j]
                after = after[1:] # 

    return after, before, common

def testExtension2(histoName, histoPrevious):
    after = "" # $histoName
    common = ""

    if '_' in histoName:
        afters = histoName.split('_')
        before = afters[0]
        nMax = len(afters)

        if ( afters[nMax - 1] == "endcaps" ):
            after = "endcaps"
            for i in range(1, nMax-1):
                before += "_" + afters[i]
        elif ( afters[nMax - 1] == "barrel" ):
            after = "barrel"
            for i in range(1, nMax-1):
                before += "_" + afters[i]
        else:
            if ( histoPrevious == "" ):
                before = histoName
                after = ""
                common = histoName
            else:
                avant =  afters[0]
                after = ""
                for i in range(1, nMax-1):
                    avant += "_" + afters[i]
                    if avant == histoPrevious:
                        before = avant
                        common = histoPrevious
                        break
                for j in range(nMax-1, nMax):
                    after += "_" + afters[j]
                after = after[1:]

    else: # no _ in histoName
        before = histoName
        common = histoName

    return [after, before, common]

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

H_da = []
n_loss = []
o_loss = []
y_pred_n = []
y_pred_o = []

data_dir = 'DATASETS' + '/{:03d}'.format(nbFiles)
data_res = 'RESULTS'
data_img = 'IMAGES'

#image_up = os.getcwd() + "/img/up.gif"
#image_point = os.getcwd() + "/img/point.gif"
image_up = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + "/img/up.gif"
image_point = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + "/img/point.gif"

#branches = ['h_ele_vertexPt', 'h_recCoreNum', 'h_recEleNum', # 0 1 2 
#            'h_recOfflineVertices', 'h_ele_chargedHadronIso', 'h_ele_etaEff_all', # 3 4 5
#            'h_ele_PoPtrueVsEta_pfx', 'h_scl_EoEtrue_barrel_new'] # 6 7
#print(branches)

branches = []
branchPath = "DATASETS/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, branchPath)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
histoKeysNames = getKeysName(tp_1, branchPath)
#print(len(histoKeysNames))

for branch in branches: # [0:8]
    fileName = data_dir + "/histo_" + branch + '_{:03d}'.format(nbFiles) + "_0_lite.txt"
    df.append(pd.read_csv(fileName))
    
nbBranches = len(branches) # [0:8]
print('there is {:03d} datasets'.format(nbBranches))

#load data from branchesHistos_NewFiles.txt file ..
fileName = data_dir + "/branchesHistos_NewFiles.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device : {:s}'.format(str(device)))

t = datetime.datetime.today()

loopMaxValue = nbBranches # nbBranches
for i in range(0, loopMaxValue):
    folderName = data_res+"/HL_1.{:03d}".format(hidden_size_1) + "_HL_2.{:03d}".format(hidden_size_2)
    if useHL3 == 1:
        folderName += "_HL_3.{:03d}".format(hidden_size_3)
    if useHL4 == 1:
        #folderName += "_HL_3.{:03d}".format(hidden_size_3) # if we kept useHL3=0, add HL3 & HL4 !
        folderName += "_HL_4.{:03d}".format(hidden_size_4)
    folderName += "_LT.{:02d}".format(latent_size) + '/' + "{:04d}".format(nbFiles)
    folderName += '/' + branches[i] + '/'
    affiche = colorText('{:03d}/{:03d}'.format(i,loopMaxValue-1), "green")
    print('\n{:s} : {:s}'.format(affiche, folderName))
    checkFolder(folderName)
            
    exportParameters = folderName + '/parameters_' + '{:s}'.format(str(branches[i]))
    exportParameters += '{:d}'.format(t.year) + '{:02d}'.format(t.month) + '{:02d}'.format(t.day) + '-'  + '{:02d}'.format(t.hour) + '{:02d}'.format(t.minute) + '.html'
    print(exportParameters)
    fParam = open(exportParameters, 'w')  # html page
    for line in createAutoEncoderRef(nbFiles, nbBranches, device, lr, epsilon, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, latent_size, batch_size, nb_epochs, percentageTrain):
        fParam.write(line)
    fParam.close()

    resumeHisto = folderName + '/histo_' + '{:s}'.format(str(branches[i]))
    resumeHisto += '{:d}'.format(t.year) + '{:02d}'.format(t.month) + '{:02d}'.format(t.day) + '-'  + '{:02d}'.format(t.hour) + '{:02d}'.format(t.minute) + '.html'
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
    fHisto.write('<b>folderName : </b>{:s}<br>\n'.format(folderName))
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

    lossesValues = data_res + "/Losses/lossesValues_" + branches[i] + ".txt"
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
    trainName = data_res + "/TrainTestLOADER/multi_train_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    testName = data_res + "/TrainTestLOADER/multi_test_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    if (useTrainLoader == 1):
        print('load %s.' % trainName)
        fHisto.write('load {:s} and {:s}<br>\n'.format(trainName, testName))
        fHisto.write('<br>\n')
        train_loader = torch.load(trainName)
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
    encoderName = folderName + "/mono_encoder_{:01d}_".format(nbLayer) + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    decoderName = folderName + "/mono_decoder_{:01d}_".format(nbLayer) + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    fHisto.write('encoderName : {:s}.<br>\n'.format(encoderName))
    fHisto.write('decoderName : {:s}.<br>\n'.format(decoderName))
    fHisto.write('<br>\n')

    lossesPictureName = folderName + '/loss_plots_' + branches[i] + "_{:03d}".format(nbFiles) + '.png'
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
            '''bo1 = train_loss < epsilon
            bo2 = test_loss < epsilon
            if (bo1 and bo2):
                break'''
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
        pictureName = folderName + '/traintestLatentPicture_' + branches[i] + '.png'
        #pictureName2 = folderName + '/traintestLatentPicture2_' + branches[i] + '.png'
        for ind in range(0, len(LatentValues_Train)):
            #print('Train ', ind, LatentValues_Train[ind])
            x_Train.append(LatentValues_Train[ind][0])
            y_Train.append(LatentValues_Train[ind][1])
            labels_Train.append(i)
        
        for ind in range(0, len(LatentValues_Test)):
            #print('Test ', ind, LatentValues_Test[ind][0])
            x_Test.append(LatentValues_Test[ind][0][0])
            y_Test.append(LatentValues_Test[ind][0][1])
            labels_Test.append(i)
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

    predLossesValues = folderName + "/predLossesValues_" + branches[i] + ".txt"
    print("loss values file : %s" % predLossesValues)
    wPred = open(predLossesValues, 'w')

    # export the y_pred_new values
    predValues = folderName + "/predValues_" + branches[i] + ".txt"
    print("values file : %s" % predValues)
    wPredVal = open(predValues, 'w')

    lossesVal = []
    latentVal = []
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
        #fHisto.write('new loss value : {:e} for {:s}<br>\n'.format(new_loss, rel))
        lossesVal.append([rel,new_loss.item()])
        latentVal.append(latent_out[0].numpy())
        #print(torch_tensor_entries_n)
        #print(y_pred_new)

        pictureName = folderName + '/predicted_new_curves_' + branches[i] + '_' + rel[6:] + '_multi.png'
        ### WARNING rel is the same for all comparisons !!!
        creatPredPictLinLog(branches[i], Ncols, torch_tensor_entries_n, y_pred_new, new_loss, rel[6:], pictureName)

        # write values into the predValues file (# export the y_pred_new values)
        text2write = rel + ',' + branches[i]
        for val in y_pred_new.numpy():
            N=len(val)
            for nn in range(0,N):
                text2write += ',' + str(val[nn])
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
    pictureName = folderName + '/comparison_loss_values_' + branches[i] + '.png'
    pictureName = folderName + '/comparison_loss_values_' + branches[i] + '.png'
    title = r"$\bf{" + branches[i] + "}$" + ' : Comparison of the losses values as function of releases.'
    createCompLossesPicture(labels,val, pictureName, title)

    title='Latent ReleasesVsTrain comparison in 2 dim'
    pictureName = folderName + '/LatentReleasesVsTrainPicture_' + branches[i] + '.svg'
    for ind, text in enumerate(labels):
        #print(text, latentVal[ind])
        x.append(latentVal[ind][0])
        y.append(latentVal[ind][1])
    createLatentPicture(labels,x,y, pictureName, title)
    pictureName = folderName + '/compLatentReleasesVsTrainPicture_' + branches[i] + '.png'
    createCompLatentPictureTrainTest(labels, x_Train,y_Train,x,y, pictureName, title)
    
    wPred.close()
    wLoss.close()
    wPredVal.close()
    print('end of %s' % branches[i])
    
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
        pictureName2 = folderName + '/predicted_new_curves_' + branches[i] + '_' + elem[0][6:] + '_multi.png'
        pictureName2 = 'https://cms-egamma.web.cern.ch/validation/Electrons/Store/AutoEncoders/' + '/' + pictureName2
        #fHisto.write('{:s}'.format(pictureName2))
        fHisto.write('<a href=\"' + pictureName2 + '\"><img width=\"250\" height=\"125\" border=\"0\" align=\"middle\" src=\"' + pictureName2 + '\"></a>')
        fHisto.write("</td></tr>\n")
    fHisto.write("\n</table>\n")
    fHisto.write("</td></tr>")

    fHisto.write('</table>')
    fHisto.close()

print('end')

