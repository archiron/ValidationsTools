#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# createFiles: create file for Kolmogorov-Smirnov maximum diff
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys
import importlib
import importlib.machinery
import importlib.util

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
else:
    print("rien")

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nKShistos")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import default as dfo
from default import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2 #, fill_Snew
from DecisionBox import DecisionBox
from sources import *

pathNb_evts = pathBase + '/' + str(NB_EVTS)
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
#folder = checkFolderName(dfo.folder)
pathCase = pathNb_evts + checkFolderName(dfo.folder)

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

rels = []
tmp_branches = []
nb_ttl_histos = []

N_histos = len(branches)
print('N_histos : %d' % N_histos)

# create folder 
if not os.path.exists(pathCase):
    try:
        os.makedirs(pathCase)
    except OSError as e:
        if e.errno != errno.EEXIST: # the folder did not exist
            raise  # raises the error again
    print('Creation of %s release folder\n' % pathCase)
else:
    print('Folder %s already created\n' % pathCase)

# get list of added ROOT files for comparison
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(pathDATA, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    #print('%s - %s' % (b[0], b[0][6:]))
    rels.append([b[0], b[0][6:], item])
    f_root = ROOT.TFile(pathDATA + item)
    h_rel = getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            print('%s OK' % branches[i])
            d = getHistoConfEntry(histo_rel)
            s_tmp = fill_Snew2(d, histo_rel)
            #s_tmp = fill_Snew(histo_rel)
            if (s_tmp.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
                tmp_branch.append('KOKO')
            elif (np.floor(s_tmp.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
                tmp_branch.append('KOKO')
            else:
                nbHistos += 1
                tmp_branch.append(branches[i])
        else:
            print('%s KO' % branches[i])
            tmp_branch.append('KOKO')
    nb_ttl_histos.append(nbHistos)
    tmp_branches.append(tmp_branch)

print('nb_ttl_histos : ', nb_ttl_histos)
if(len(set(nb_ttl_histos))==1):
    print('All elements are the same with value {:d}.'.format(nb_ttl_histos[0]))
else:
    print('All elements are not the same.')
    print('nb ttl of histos : ' , nb_ttl_histos)
newBranches = optimizeBranches(tmp_branches)

if (len(branches) != len(newBranches)):
    print('len std branches : {:d}'.format(len(branches)))
    print('len new branches : {:d}'.format(len(newBranches)))
    branches = newBranches
    N_histos = len(branches)
print('N_histos : %d' % N_histos)

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathNb_evts, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
checkFolder(pathNb_files)

#print('-')
#for elem in rels:
#    print(elem)
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

for i in range(0, N_histos): #, N_histos-1 1 N_histos histo for debug
    print(branches[i]) # print histo name

    for elem in sortedRels:
        rel = elem[1]
        KSDiffHistoName1 = pathNb_files + '/KSDiffHistoValues_1_' + branches[i] + "_" + rel + '.txt'
        KSDiffHistoName3 = pathNb_files + '/KSDiffHistoValues_3_' + branches[i] + "_" + rel + '.txt'
        #print('KSDiffHistoName 1 : {:s}'.format(KSDiffHistoName1))
        #print('KSDiffHistoName 3 : {:s}'.format(KSDiffHistoName3))

        wKSDiff1 = open(KSDiffHistoName1, 'r')
        Lines1 = wKSDiff1.readlines()
        wKSDiff1.close()
        ord1 = Lines1[0].split()
        ord1 = np.asarray(ord1).astype(float)
        abs1 = Lines1[1].split()
        abs1 = np.asarray(abs1).astype(float)
        abs11 = []
        for j in range(0, len(abs1)-1):
            abs11.append((abs1[j]+abs1[j+1])/2)
        wKSDiff3 = open(KSDiffHistoName3, 'r')
        Lines3 = wKSDiff3.readlines()
        wKSDiff3.close()
        ord3 = Lines3[0].split()
        ord3 = np.asarray(ord3).astype(float)
        abs3 = Lines3[1].split()
        abs3 = np.asarray(abs3).astype(float)
        abs31 = []
        for j in range(0, len(abs3)-1):
            abs31.append((abs3[j]+abs3[j+1])/2)
        
        longu = len(abs11)
        ord_1 = []
        abs_1 = []
        for j in range(0, longu-1):
            if ( ord1[j] != 0. ):
                abs_1.append(abs11[j])
                ord_1.append(ord1[j])

        longu = len(abs31)
        ord_3 = []
        abs_3 = []
        for j in range(0, longu-1):
            if ( ord3[j] != 0. ):
                abs_3.append(abs31[j])
                ord_3.append(ord3[j])
        
        longu = len(abs_1)
        x_cent1 = 0.
        for j in range(0, longu-1):
            x_cent1 += abs_1[j] * ord_1[j]
        x_cent1 /= sum(ord_1)
        dx_cent1 = 0.
        for j in range(0, longu-1):
            dx_cent1 += (abs_1[j] - x_cent1) * (abs_1[j] - x_cent1) * ord_1[j]
        dx_cent1 = sqrt(dx_cent1/sum(ord_1))
        longu = len(abs_3)
        x_cent3 = 0.
        for j in range(0, longu-1):
            x_cent3 += abs_3[j] * ord_3[j]
        x_cent3 /= sum(ord_3)
        dx_cent3 = 0.
        for j in range(0, longu-1):
            dx_cent3 += (abs_3[j] - x_cent3) * (abs_3[j] - x_cent3) * ord_3[j]
        dx_cent3 = sqrt(dx_cent3/sum(ord_3))
        #print('x_cent 1 : {:f}'.format(x_cent1))
        #print('dx_cent 1 : {:f}'.format(dx_cent1))
        #print('x_cent 3 : {:f}'.format(x_cent3))
        #print('dx_cent 3 : {:f}'.format(dx_cent3))

        cx1 = []
        cy1 = []
        cx3 = []
        cy3 = []
        cx1.append(x_cent1 - dx_cent1)
        cx1.append(x_cent1 + dx_cent1)
        cx3.append(x_cent3 - dx_cent3)
        cx3.append(x_cent3 + dx_cent3)
        cy1.append(max(ord_1)/2.)
        cy1.append(max(ord_1)/2.)
        cy3.append(max(ord_3)/2.)
        cy3.append(max(ord_3)/2.)

        HistoFileName = pathNb_files + '/KSCompHisto_' + branches[i] + "_" + rel + '.png'
        plt.clf()
        plt.figure(figsize=(10, 5))
        fig, ax1 = plt.subplots()
        ax1.set_title(rel, x=0.50, y=1.05)
        #plot_1 = ax1.plot(abs_1, ord_1, color='blue', marker='*', linestyle = 'None', label='KS 1')
        plot_1 = ax1.step(abs_1, ord_1, where='mid', color='blue', marker='*', label='KS 1')
        #plot_1 = ax1.stairs(abs_1, ord_1, fill=True, color='blue', marker='*', label='KS 1')
        ax1.plot(cx1, cy1, color='blue', marker='*', linewidth=2)
        ax2 = ax1.twinx()
        #plot_2 = ax2.plot(abs_3, ord_3, color='green', marker='+', linestyle = 'None', label='KS 3')
        plot_2 = ax2.step(abs_3, ord_3, where='mid', color='red', marker='+', label='KS 3')
        ax2.plot(cx3, cy3, color='red', marker='+', linewidth=2)
        lns = plot_1 + plot_2
        labels2 = [l.get_label() for l in lns]
        plt.legend(lns, labels2, loc=0)
        plt.tight_layout()
        plt.savefig(HistoFileName)

print("Fin !\n")
