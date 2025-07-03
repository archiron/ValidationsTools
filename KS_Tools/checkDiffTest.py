#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# createFiles: create file for Kolmogorov-Smirnov maximum diff
# for egamma validation comparison                              
#
# create the Kolmogorov curves.
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys,shutil
import importlib
import importlib.machinery
import importlib.util
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True

#from ROOT import gROOT
root_version = ROOT.gROOT.GetVersion()

from numpy.random import rand
import pandas as pd
import numpy as np
import matplotlib

print('PANDAS     version : {}'.format(pd.__version__))
print('PYTHON     version : {}'.format(sys.version))
print("NUMPY      version : {}".format(np.__version__))
print('MATPLOTLIB version : {}'.format(matplotlib.__version__))
print("ROOT       version : {}".format(root_version))

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # FileName for paths
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
else:
    print("rien")
    pathBase = ''

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\ncreateFiles_v4")

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

import validationsDefault as dfo
from validationsDefault import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import Graphic
from graphicAutoEncoderFunctions import GraphicKS
from DecisionBox import DecisionBox
from filesSources import *

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathROOTFiles = blo.pathROOT + "/" + release
pathROOTFiles = checkFolderName(pathROOTFiles)
print('pathROOTFiles : {:s}'.format(pathROOTFiles))

DB = DecisionBox()
grKS = GraphicKS()
gr = Graphic()
gr.initRoot()

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

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathROOTFiles) # get the list of the root files in the folderName folder
if (len(rootFilesList_0) ==0 ):
    print('there is no generated ROOT files')
    exit()
else:
    print('there is {:d} generated ROOT files'.format(len(rootFilesList_0)))
rootFilesList_0.sort()
print('there is %d generated ROOT files' % len(rootFilesList_0))
rootFilesList_0 = rootFilesList_0[0:nbFiles]
#print('file list :')
#print(rootFilesList_0)
nbFiles = 2000 # change_nbFiles(len(rootFilesList_0), nbFiles)

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
checkFolder(pathNb_files)
pathKS = pathNb_files + 'KS'
pathKS =checkFolderName(pathKS)
checkFolder(pathKS)
print('pathNb_files : {:s}'.format(pathNb_files))
print('pathKS : {:s}'.format(pathKS))

# get list of added ROOT files for comparison
pathDATA = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Run3/RECO/') # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
print('pathDATA : {:s}'.format(pathDATA))
rootFilesList = getListFiles(pathDATA, 'root')
rootFilesList2 = []
rootList2 = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Values2/rootSourcesRelValZEE_14mcRun3RECO/rootSourcesRelValZEE_14mcRun3RECO.txt')
sourceList = open(rootList2, "r")
for ligne in sourceList:
    t_ligne = ligne.replace('_0.txt', '.root')
    t_ligne = t_ligne.replace('_1.txt', '.root')
    #print('[{:s}] - [{:s}]'.format(ligne, t_ligne))
    rootFilesList2.append(t_ligne.rstrip())
compteur = 0
for item in rootFilesList2:
    print('\n{:2d} : {:s}'.format(compteur, item))
    compteur += 1
rootFilesList3 = []
for item in rootFilesList2: 
    if item not in rootFilesList3: 
        rootFilesList3.append(item) 
compteur = 0
for item in rootFilesList3:
    print('\n{:2d} : {:s}'.format(compteur, item))
    compteur += 1
print('we use the files :')
for item in rootFilesList3:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    name = os.path.join(pathDATA, item)
    print('{:s} : {:d}'.format(item, len(item)))
    print('name : {:s}'.format(name))
    f_root = ROOT.TFile(name)
    h_rel = gr.getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            d = gr.getHistoConfEntry(histo_rel)
            s_tmp = gr.fill_Snew2(d, histo_rel)
            if (s_tmp.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
            elif (np.floor(s_tmp.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
            else:
                nbHistos += 1
                tmp_branch.append(branches[i])
        else:
            print('%s KO' % branches[i])
    nb_ttl_histos.append(nbHistos)
    tmp_branches.append(tmp_branch)

#print('nb_ttl_histos : ', nb_ttl_histos)
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
branches2 = sorted(branches)
print(branches2)
branches = branches2

print('pathNb_files : {:s}'.format(pathNb_files))
source_dest = pathNb_files + "/ElectronMcSignalHistos.txt"
print('source : {:s}'.format(source))
print('source_dest : {:s}'.format(source_dest))
shutil.copy2(source, source_dest)

sortedRels = rels # sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

# get the "reference" root file datas
if (ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)
input_ref_file = pathROOTFiles + '/DQM_V0001_R000000001__RelValZEE_14__CMSSW_14_1_0__RECO_9000_' + '{:03d}'.format(ind_reference) + ".root"
print('input_ref_file : {:s}'.format(input_ref_file))

f_KSref = ROOT.TFile(input_ref_file) # pathDATA + 
print('we use the %s file as KS reference' % input_ref_file)
h_KSref = gr.getHisto(f_KSref, tp_1)

nbRels = len(sortedRels)

tic = time.time()

pV_tab = []
for i in range(0, N_histos):#,N_histos, N_histos-1 range(N_histos - 1, N_histos):  # 1 N_histos histo for debug
    print('histo : {:s}'.format(branches[i])) # print histo name
    
    histo_KSref = h_KSref.Get(branches[i])
    if (histo_KSref):
        print('%s OK' % branches[i])
    
        histo_KSref = h_KSref.Get(branches[i])
        s_KSref = []
        for entry in histo_KSref:
            s_KSref.append(entry)
        s_KSref = np.asarray(s_KSref)
        s_KSref = s_KSref[1:-1]
        print(s_KSref)

        # get nb of columns & rows for histos
        Ncols = len(s_KSref)
        print('[Ncols] : [%d]' % (Ncols))

        s0 = np.asarray(s_KSref) # if not this, ind is returned as b_00x instead of int value
        s1 = np.asarray(s_KSref[1:])
        N0 = len(s0)
        s1 = np.append(s1, [s0[N0-1]]) ## arbitrary decision !!
        N1 = len(s1)
        #print(s0)
        #print(s1)
        if (N0 != N1):
            print('not the same lengths')
            exit()
        min0 = min(s0)
        min1 = min(s1)
        min01 = min(min0, min1)
        if (min01 > 0.):
            min01 = 0.
        else:
            min01 = np.abs(min01)
        #print('min01 : {}'.format(min01))
        SumSeries0 = s0.sum() + N0 * min01
        SumSeries1 = s1.sum() + N1 * min01
        v0 = 0.
        v1 = 0.
        sDKS = []
        for j in range(0, N0):
            t0 = (min01 + s0[j])/SumSeries0
            t1 = (min01 + s1[j])/SumSeries1
            v0 += t0
            v1 += t1
            sDKS.append(np.abs(v1 - v0))
        v = max(sDKS)
        #print(v0)
        #print(v1)
        #print(sDKS)

        b = np.random.randn(N0)
        s2 = s0 * (1 + 0.1 * b)
        #print(s2)
        min0 = min(s0)
        min2 = min(s2)
        min02 = min(min0, min2)
        if (min02 > 0.):
            min02 = 0.
        else:
            min02 = np.abs(min02)
        SumSeries0 = s0.sum() + N0 * min01
        SumSeries2 = s2.sum() + N0 * min02
        v0 = 0.
        v2 = 0.
        sDKS2 = []
        for j in range(0, N0):
            t0 = (min01 + s0[j])/SumSeries0
            t2 = (min02 + s2[j])/SumSeries2
            v0 += t0
            v2 += t2
            sDKS2.append(np.abs(v2 - v0))
        v2 = max(sDKS2)
        #print(v0)
        #print(v2)
        #print(sDKS2)

        # draw the picture with KS plot and diff position
        fileName1 = pathKS + '/KS-ttlDiff_1_' + branches[i] + '_v5.png'
        grKS.createSimpleDiffPictur2(branches[i] + ' : ' + str(N0), sDKS, sDKS2, ['bins', 'norm. diff.'], ['bit diff', 'random diff'], fileName1)
        
    else:
        print('%s KO' % branches[i])

#Stop()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !\n")

