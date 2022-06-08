#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils import data
from torch.nn.functional import normalize
import os
import pandas as pd

from autoEncoders import *
from graph import *
from controlFunctions import *
from defaultStd import *

df = []
df_entries = []
df_errors = []

torch_tensor_entries = []
torch_tensor_errors = []

train_loader = []
test_loader = []

data_dir = 'DATASETS'
data_res = 'RESULTS'
data_img = 'IMAGES'

branches = ['h_ele_vertexPt', 'h_recCoreNum', 'h_recEleNum', # 0 1 2 
            'h_recOfflineVertices', 'h_ele_chargedHadronIso', 'h_ele_etaEff_all', # 3 4 5
            'h_ele_PoPtrueVsEta_pfx', 'h_scl_EoEtrue_barrel_new'] # 6 7
print(branches)
branch = branches[iDisplay]
fileName = data_dir + "/histo_" + branch + '_{:03d}'.format(nbFiles) + "_0_lite.txt"
print('%s' % fileName)
df=pd.read_csv(fileName)

lossesValues = data_res + "/lossesValues_" + branch + ".txt"
print("loss values file : %s" % lossesValues)
wLoss = open(lossesValues, 'w')

#print(df.head(5))

cols = df.columns.values
n_cols = len(cols)
print('nb of columns for histo {:s} : {:d}'.format(branch, n_cols))
cols_entries = cols[6::2]
cols_errors = cols[7::2]
df_entries=df[cols_entries]
df_errors=df[cols_errors]

#print(df_entries.head(5))
#print('---')
#print(df_errors.head(5))
#print('')

# get nb of columns & rows for histos & remove over/underflow
(Nrows, Ncols) = df_entries.shape
print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branch))
df_entries = df_entries.iloc[:, 1:Ncols-1]
(Nrows, Ncols) = df_entries.shape
print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branch))

#(Nrows, Ncols) = df_errors.shape
#print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branch))
#df_errors = df_errors.iloc[:, 1:Ncols-1]
#(Nrows, Ncols) = df_errors.shape
#print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branch))

#load data from branchesHistos_RelRef.txt file ..
fileName = data_dir + "/branchesHistos_RelRef.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()

new = []
old = []

i = 0
for line in Lines:
    a,b = line.rstrip().split(',', 1)
    a = a[4:]
    if ( str(a) == str(branch)):
        #print(a,':',b) # remove new/old
        break
    i+=1

_, line = Lines[i].rstrip().split(',', 1)
new = line.rstrip().split(',')
new = np.asarray(new).astype(float)

_, line = Lines[i+1].rstrip().split(',', 1)
old = line.rstrip().split(',')
old = np.asarray(old).astype(float)

print(new)
#print(old)

# .. and plot them
pictureName = data_img + '/newold_steps_' + branch + "_{:03d}".format(nbFiles)  + '.png'
createCompPicture(branch, Ncols, new, old, "new", "old", pictureName)
print('')

#load data from branchesHistos_NewFiles.txt file ..
fileName = data_dir + "/branchesHistos_NewFiles.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()
linOp = []

i = 0
for line in Lines:
    rel,b = line.rstrip().split(',', 1)
    hName = b.rstrip().split(',', 1)[0]
    if ( str(hName) == str(branch)):
        print('equal === ',rel,':',hName) # remove new/old
        linOp.append(line)
    i+=1

for elem in linOp:
    print(elem)
#stop

trainName = data_res + "/TrainTestLOADER/mono_train_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
testName = data_res + "/TrainTestLOADER/mono_test_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
if (useTrainLoader == 1):
    train_loader = torch.load(trainName)
    test_loader = torch.load(testName)
    print('load OK.')
else:
    # creating torch tensor from df_entries/errors
    torch_tensor_entries=torch.tensor(df_entries.values)
    print('max df')
    print(df_entries.values.max())
    MAXMax = df_entries.values.max()
    print('MAXMax : %e' % MAXMax)
    #torch_tensor_errors=torch.tensor(df_errors.values)
    print(torch_tensor_entries)
    #print(torch_tensor_errors)

    # normalize the tensor
    torch_tensor_entries_n = normalize(torch_tensor_entries, p=2.0)
    print(torch_tensor_entries_n)
    print('torch_tensor_entries_n')
    print(torch_tensor_entries.shape)

    train_size=int(percentageTrain*len(torch_tensor_entries_n)) # in general torch_tensor_entries[i] = 200
    test_size=len(torch_tensor_entries_n)-train_size
    print('train size : %d' % train_size)
    print('test size  : %d' % test_size)
    train_tmp, test_tmp = data.random_split(torch_tensor_entries_n,[train_size,test_size])

    train_loader = data.DataLoader(train_tmp,batch_size=batch_size)
    test_loader = data.DataLoader(test_tmp,batch_size=batch_size)
        
    print('saving ...')
    torch.save(train_loader,trainName)
    torch.save(test_loader,testName)
    print('save OK.')

# Ready for initialization

(_, Ncols) = df_entries.shape

loss_fn=torch.nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device : %s' % device)

Ncols2 = int(Ncols/4)
LTsize = int(Ncols/10)
for i1 in range(1, N_HL1): #6
    for i2 in range(1, N_HL2):#5
        for i3 in range(1, N_HL3):#5
            hidden_size_1 = i1 * Ncols
            hidden_size_2 = i2 * Ncols2
            latent_size = i3 * LTsize
            folderName = data_res + "/HL_1.{:03d}".format(hidden_size_1) + "_HL_2.{:03d}".format(hidden_size_2)
            folderName += "_LT.{:02d}".format(latent_size) + '/' + branch + '/'
            #print('\n'+folderName)
            checkFolder(folderName)
            
            t1 = 100*i1/(N_HL1-1)
            t2 = 100*i2/(N_HL2-1)
            t3 = 100*i3/(N_HL3-1)
            evol = colorText("{0:4.2f}".format(t1), "lightyellow") + "_" + colorText("{0:4.2f}".format(t2), "green") + "_" + colorText("{0:4.2f}".format(t3), "blue")
            print('=================================')
            #print("HIDDEN SIZE 1 : {:d}, HIDDEN SIZE 2 : {:d}, LATENT SIZE : {:d}".format(hidden_size_1,hidden_size_2,latent_size) )
            print(colorText("HIDDEN SIZE 1 : {:d}".format(hidden_size_1), "lightyellow") + ", " + colorText("HIDDEN SIZE 2 : {:d}".format(hidden_size_2), "green") + ", " + colorText("LATENT SIZE : {:d}".format(latent_size), "blue") )
            print("\t\t" + evol)
            print('=================================')
            
            #define the network
            encoder=Encoder2(latent_size,Ncols,hidden_size_1,hidden_size_2)
            decoder=Decoder2(latent_size,Ncols,hidden_size_1,hidden_size_2)
            #encoder=Encoder4(latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
            #decoder=Decoder4(latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)

            params_to_optimize=[
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
            ]

            optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)

            history_da={'train_loss':[],'test_loss':[]}
            L_out = []

            # Ready for calculation
            encoderName = folderName + "/mono_encoder_" + branch + "_{:03d}".format(nbFiles) + ".pth"
            decoderName = folderName + "/mono_decoder_" + branch + "_{:03d}".format(nbFiles) + ".pth"

            if ( useEncoder == 1):
                if not os.path.isfile(encoderName):
                    print('%s does not exist' % encoderName)
                    exit()
                else:
                    encoder = torch.load(encoderName)
                if not os.path.isfile(decoderName):
                    print('%s does not exist' % decoderName)
                    exit()
                else:
                    decoder = torch.load(decoderName)
            else:
                for epoch in range(nb_epochs):
                    train_loss=train_epoch_den(encoder=encoder, decoder=decoder,device=device,
                        dataloader=train_loader, loss_fn=loss_fn,optimizer=optim)
                    test_loss, d_out=test_epoch_den(encoder=encoder, decoder=decoder,device=device,
                        dataloader=test_loader, loss_fn=loss_fn)
                    L_out.append(d_out)
                    #r = (train_loss - test_loss) / (train_loss + test_loss)
                    #print('epoch : %d : tr_lo = %e : te_lo = %e' % (epoch, train_loss, test_loss, r))
                    history_da['train_loss'].append(train_loss)
                    history_da['test_loss'].append(test_loss)
                    '''bo1 = train_loss < epsilon
                    bo2 = test_loss < epsilon
                    if (bo1 and bo2):
                        break'''
                r = (train_loss - test_loss) / (train_loss + test_loss)
                print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
                #print('epoch : %d : tr_lo = %e : te_lo = %e' % (epoch, train_loss, test_loss))
                torch.save(encoder,encoderName)
                torch.save(decoder,decoderName)

                #print('write HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e'
                #            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))
                wLoss.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
                            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))

                lossesPictureName = folderName + '/loss_plots_' + branch + "_{:03d}".format(nbFiles) + '.png'
                #createLossPictures(branch, history_da, nb_epochs, lossesPictureName)
                createLossPictures(branch, history_da, epoch+1, lossesPictureName)

            # Ready for prediction
            print('using %s\n' % encoderName)
            #encoder = torch.load(encoderName)
            #decoder = torch.load(decoderName)

            predLossesValues = folderName + "/predLossesValues_"+branch+".txt"
            print("loss values file : %s" % predLossesValues)
            wPred = open(predLossesValues, 'w')

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

                print('using mono_encoder_'+branch+'.pth\n')
                encoder = torch.load(encoderName)
                decoder = torch.load(decoderName)

                ## TEMP TEMP (bof ne semble pas jouer)
                loss_fn=torch.nn.MSELoss()

                params_to_optimize=[
                {'params': encoder.parameters()},
                {'params': decoder.parameters()}
                ]

                optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)

                # Forward pass: Compute predicted y by passing x to the model
                new_loss, y_pred_new = test_epoch_den(encoder=encoder,
                        decoder=decoder,device=device,
                        dataloader=test_loader_n,
                        loss_fn=loss_fn)

                # Compute and print loss
                print('new loss value : %e for %s' % (new_loss, rel))
                wPred.write('%e, %s\n' % (new_loss, rel))
                #print(torch_tensor_entries_n)
                #print(y_pred_new)
                #print('')

                pictureName = folderName + '/predicted_diff_curves_' + branch + '_' + rel[6:] + '.png'
                creatPredPictLinLog(branch, Ncols, torch_tensor_entries_n, y_pred_new, new_loss, rel, pictureName)

            wPred.close()
            #stop
    
wLoss.close()
print('End')
