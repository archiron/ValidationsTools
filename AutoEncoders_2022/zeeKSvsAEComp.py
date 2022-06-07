#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeeKSvsAEComp: create a KS comparison (max diff) between the original curve 
# and the predicted one for different egamma validation releases.
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

from genericpath import exists
import os,sys
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('agg')

#Chilib_path = '/pbs/home/c/chiron/private/KS_Tools/ChiLib_CMS_Validation'
Chilib_path = '/home/arnaud/cernbox/DEV_PYTHON/ChiLib'
sys.path.append(Chilib_path)
import default as dfo
from default import *
from defaultStd import *
from sources import *
from graph import *
from controlFunctions import *

from DecisionBox import *
DB = DecisionBox()

# these line for daltonians !
#seaborn.set_palette('colorblind')

def func_createKSvsAECompare(branches):
    rels = []

    print("func_createKSvsAECompare")
    dfo.folderName = checkFolderName(dfo.folderName)
    dfo.folder = checkFolderName(dfo.folder)
    
    data_dir = 'DATASETS' + '/{:03d}'.format(nbFiles)
    data_res = 'RESULTS'

    N_histos = len(branches)
    print('N_histos : %d' % N_histos)
    
    # create folder 
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST: # the folder did not exist
                raise  # raises the error again
        print('Creation of %s release folder\n' % folder)
    else:
        print('Folder %s already created\n' % folder)

    # get list of files
    LOG_SOURCE_WORK = 'DATASETS/NewFiles/'
    rootFilesList = getListFiles(LOG_SOURCE_WORK, 'root')
    print('we use the files :')
    for item in rootFilesList:
        print('%s' % item)
        b = (item.split('__')[2]).split('-')
        rels.append([b[0], b[0][6:]])
    labels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

    #load data from branchesHistos_NewFiles.txt file ..
    fileName = data_dir + "/branchesHistos_NewFiles.txt"
    print('%s' % fileName)
    file1 = open(fileName, 'r')
    Lines = file1.readlines()
    #for l in range(0,13):
    #     print('{:02d}/{:02d} : {:s}'.format(l,12,Lines[l][:-1]))
    file1.close()

    tic = time.time()

    nbBranches = len(branches) # [0:8]
    print('there is {:03d} datasets'.format(nbBranches))

    loopMaxValue = nbBranches # nbBranches
    for nb in range(0, loopMaxValue):
        #print(branches[nb])

        # create the folder name
        folderName = data_res+"/HL_1.{:03d}".format(hidden_size_1) + "_HL_2.{:03d}".format(hidden_size_2)
        if useHL3 == 1:
            folderName += "_HL_3.{:03d}".format(hidden_size_3)
        if useHL4 == 1:
            folderName += "_HL_4.{:03d}".format(hidden_size_4)
        folderName += "_LT.{:02d}".format(latent_size) + '/' + "{:04d}".format(nbFiles)
        folderName += '/' + branches[nb] + '/'
        affiche = colorText('{:03d}/{:03d}'.format(nb,loopMaxValue-1), "green")
        print('\n{:s} : {:s}'.format(affiche, folderName))
        checkFolder(folderName)

        # read the predicted values
        predValues = folderName + "/predValues_" + branches[nb] + ".txt"
        #print("values file : %s" % predValues)
        wPredVal = open(predValues, 'r')
        LinesPred = wPredVal.readlines()
        #for l in range(0,len(LinesPred)):
        #    print('{:02d}/{:02d} : {:s}'.format(l,12,LinesPred[l]))
        wPredVal.close()

        # extract the values of originals curves from Lines
        LinesOrig = []
        for l in Lines:
            l_split = l.split(',')[1]
            #print(l_split)
            if ( l_split == branches[nb] ):
                LinesOrig.append(l)
        #for l in range(0,len(LinesOrig)):
        #    print('{:02d}/{:02d} : {:s}'.format(l,12,LinesOrig[l]))

        print('len LinesPred : {}'.format(len(LinesPred)))
        print('len LinesOrig : {}'.format(len(LinesOrig)))
        #print(LinesPred.shape)

        labels1 = []
        val1 = []
        for ll in labels:
            predKSValues = 'DATASETS/KS' + '/{:03d}'.format(nbFiles) + "/histo_differences_KScurve_" + ll[1] + "__{:03d}".format(nbFiles) + ".txt"
            #print("values file : %s" % predKSValues)
            wPredKSVal = open(predKSValues, 'r')
            LinesKSVal = wPredKSVal.readlines()
            #for l in range(0,len(LinesKSVal)):
            #    print('{:02d}/{:02d} : {:s}'.format(l,259,LinesKSVal[l]))
            wPredKSVal.close()
            labels1.append(ll[1])
            for l in range(0, len(LinesKSVal)):
                l_pred = LinesKSVal[l][:-1].split(' : ')
                if ( l_pred[0] == branches[nb] ):
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
            pictureName = folderName + 'comparison_KSvsAE_' + branches[nb] + '_{:03d}'.format(nbFiles) +'.png' # 
            #print(pictureName)
            title = r"$\bf{" + branches[nb] + "}$" + ' : Comparison of KS vs AE values as function of releases.'
            #print(title)
            createCompKSvsAEPicture(labels2, val1, val2, pictureName, title)
            pictureName = folderName + 'comparison_KSvsAE_2Axis_' + branches[nb] + '_{:03d}'.format(nbFiles) +'.png' # 
            createCompKSvsAEPicture2Axis(labels2, val1, val2, pictureName, title)

    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
    branches = getBranches(tp_1, source)
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    func_createKSvsAECompare(branches)  # create the KS files from histos datas

    print("Fin !")

