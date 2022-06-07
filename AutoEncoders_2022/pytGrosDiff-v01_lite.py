#!/usr/bin/env python
# coding: utf-8

from matplotlib.transforms import BboxBase
import numpy as np
import torch
from torch.utils import data
from torch.nn.functional import normalize
import os
import pandas as pd

from autoEncoders import *
from graph import *
from functions import *
from defaultDiff import *

def calculateSecondHistoFill(n1,n2):
    N = len(n1)
    second = []
    # calculate a second array by mixing mean and new
    for i in range(0,N):
        a = i % 2
        if (a == 0):
            second.append(n1[i])
        else:
            second.append(n2[i])
    second = np.asarray(second)
    return second

df = []
df_entries = []
df_errors = []

torch_tensor_entries = []
torch_tensor_errors = []

train_loader = []
test_loader = []

linOp = []

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

lossesValues = data_res + "/lossesDiffValues_"+branch+".txt"
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

# load data from branchesHistos_NewFiles.txt file .. #
fileName = data_dir + "/branchesHistos_NewFiles.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()

i = 0
for line in Lines:
    rel,b = line.rstrip().split(',', 1)
    hName = b.rstrip().split(',', 1)[0]
    if ( str(hName) == str(branch)):
        print(' === ',rel,':',hName) # remove new/old
        linOp.append(line)
    i+=1

trainName = data_res + "/TrainTestLOADER/mono_trainDiff_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
testName = data_res + "/TrainTestLOADER/mono_testDiff_loader_" + branch + "_{:03d}".format(nbFiles) + ".pth"
if (useTrainLoader == 1):
    train_loader = torch.load(trainName)
    test_loader = torch.load(testName)
    print('load OK.')
else:
    # =============================== #
    # create the array of differences #
    # =============================== #
    nb1 = 0
    totalDiff = []
    for k in range(0,Nrows-1):
        for l in range(k+1, Nrows):
            nb1 += 1
            series0 = df_entries.iloc[k,:]
            series1 = df_entries.iloc[l,:]     
            totalDiff.append(series0 - series1)
            if ((nb1 % 10) == 0):
                print(nb1, end='\r')

    print('ttl nb1 of couples : %d' % nb1)
    #print(totalDiff)
    #print(np.asarray(totalDiff))
    totalDiff = np.asarray(totalDiff)

    # creating torch tensor from df_entries/errors
    torch_tensor_entries = torch.tensor(totalDiff)
    print('max totalDiff')
    print(totalDiff.max())
    MAXMax = totalDiff.max()
    print('MAXMax : %e' % MAXMax)
    print(torch_tensor_entries)
    print('max torch_tensor_entries : %f' % torch.max(torch_tensor_entries))

    # normalize the tensor
    torch_tensor_entries_n = normalize(torch_tensor_entries, p=2.0)
    #print(torch_tensor_entries_n)
    print('torch_tensor_entries_n')
    #print(torch.max(torch_tensor_entries_n))
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

folderName = data_res+"/HL_1.{:03d}".format(hidden_size_1) + "_HL_2.{:03d}".format(hidden_size_2)
folderName += "_LT.{:02d}".format(latent_size) + '/' + branch + '/'
print('\n'+folderName)
checkFolder(folderName)

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
encoderName = folderName + "/mono_encoderDiff_" + branch + "_{:03d}".format(nbFiles) + ".pth"
decoderName = folderName + "/mono_decoderDiff_" + branch + "_{:03d}".format(nbFiles) + ".pth"

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
        r = (train_loss - test_loss) / (train_loss + test_loss)
        print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
        history_da['train_loss'].append(train_loss)
        history_da['test_loss'].append(test_loss)
        bo1 = train_loss < epsilon
        bo2 = test_loss < epsilon
        if (bo1 and bo2):
            break
    print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
    torch.save(encoder,encoderName)
    torch.save(decoder,decoderName)

    wLoss.write('HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e\n' 
                % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))

    lossesPictureName = folderName + '/loss_plots_' + branch + "_{:03d}".format(nbFiles) + '.png'
    #createLossPictures(branch, history_da, nb_epochs, lossesPictureName)
    createLossPictures(branch, history_da, epoch+1, lossesPictureName)

#stop
# Ready for prediction
print('using %s\n' % encoderName)

predLossesValues = folderName + "/predDiffLossesValues_"+branch+".txt"
print("loss values file : %s" % predLossesValues)
wPred = open(predLossesValues, 'w')

nbReleases = len(linOp)
print('there is %d releases to compare' % nbReleases)
totalDiff1 = np.zeros((nbReleases,nbReleases))
xlist = np.linspace(0, nbReleases-1, nbReleases)
X, Y = np.meshgrid(xlist, xlist)
listRel = []

for k in range(0, nbReleases-1):
    for l in range(k+1, nbReleases):
        rel1, hName1, line1 = linOp[k].rstrip().split(',', 2)
        rel2, hName2, line2 = linOp[l].rstrip().split(',', 2)
        print('### %s / %s ###' % (rel1, rel2))
        print('[%d,%d] : %s %s %s %s'%(k, l, rel1, hName1, rel2,hName2))
        new1 = line1.rstrip().split(',')
        new1 = np.asarray(new1).astype(float)
        new2 = line2.rstrip().split(',')
        new2 = np.asarray(new2).astype(float)
        #print('max : %f' % np.amax(new2))
        pictureName = folderName + '/newold_steps_' + branch + '_' + rel1[6:] + '-' + rel2[6:] + '.png'
        createCompPicture(branch, Ncols, new1, new2, rel1[6:], rel2[6:], pictureName)
        new_diff = new1 - new2
        df_new = pd.DataFrame(new_diff).T # otherwise, one column with 50 rows instead of 1 line with 50 columns
        # creating torch tensor from df_entries/errors
        torch_tensor_new = torch.tensor(df_new.values)

        # normalize the tensor
        torch_tensor_entries_n = normalize(torch_tensor_new, p=2.0)
        #print(torch_tensor_entries_n)

        test_loader_n = data.DataLoader(torch_tensor_entries_n)

        encoder = torch.load(encoderName)
        decoder = torch.load(decoderName)
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
        text = 'new loss value : '+colorText(str(new_loss.numpy()), 'lightyellow')+' for rel1 vs rel2'
        print(text)
        wPred.write('%e, %s, %s\n' % (new_loss, rel1, rel2))
        #print(torch_tensor_entries_n)
        #print(y_pred_new)

        pictureName = folderName + '/predicted_diff_curves_' + branch + '_' + rel1[6:] + '-' + rel2[6:] + '.png'
        creatPredPictLin(branch, Ncols, torch_tensor_entries_n, y_pred_new, new_loss, rel1[6:], pictureName)

        totalDiff1[k][l] = new_loss
    listRel.append(rel1[6:])

listRel.append(rel2[6:])
pictureName = folderName + '/map-ttlDiff_' + branch + '_' + rel1[6:] + '-' + rel2[6:] + '.png'
createMapPicture(X, Y, totalDiff1, listRel, pictureName)

wPred.close()
#stop
    
wLoss.close()
print('End')
