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

from ROOT import kWhite, kBlue, kBlack, kRed, gStyle, TCanvas, gPad
root_version = ROOT.gROOT.GetVersion()

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

#import pandas as pd
import numpy as np
import matplotlib

#print('PANDAS     version : {}'.format(pd.__version__))
print('PYTHON     version : {}'.format(sys.version))
print("NUMPY      version : {}".format(np.__version__))
print('MATPLOTLIB version : {}'.format(matplotlib.__version__))
print("ROOT      version : {}".format(root_version))

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nKSvalidation-ter")

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
from graphicFunctions import *
from graphicFunctions import Graphic
from DecisionBox import DecisionBox 
from filesSources import *
from fonctions import Tools
from valEnv_default import env_default

'''def shortHistoName(elem):
    histo_names = elem.split("/")
    #histo_name = histo_names[0]
    histoShortNames = histo_names[1]
    histo_pos = histoShortNames
    histo_positions = histo_pos.split()
    short_histo_names = histoShortNames.split(" ")
    short_histo_name = short_histo_names[0].replace("h_", "")
    if "ele_" in short_histo_name:
        short_histo_name = short_histo_name.replace("ele_", "")
    if "scl_" in short_histo_name:
        short_histo_name = short_histo_name.replace("scl_", "")
    if "bcl_" in short_histo_name:
        short_histo_name = short_histo_name.replace("bcl_", "")
    return short_histo_name, short_histo_names, histo_positions'''

sys.path.append(os.getcwd()) # path where you work

DB = DecisionBox()
tl = Tools()
gr = Graphic()
valEnv_d = env_default()

gr.initRoot()

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = pathChiLib + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathCase = pathNb_evts + checkFolderName(dfo.folder)
print('pathCase : {:s}'.format(pathCase))
pathROOTFiles = blo.pathROOT + "/" + release
pathROOTFiles = checkFolderName(pathROOTFiles)
print('pathROOTFiles : {:s}'.format(pathROOTFiles))

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
#rootFilesList_0 = getListFiles(pathROOTFiles) # get the list of the root files in the folderName folder
#if (len(rootFilesList_0) ==0 ):
#    print('there is no generated ROOT files')
#    exit()
#rootFilesList_0.sort()
#print('there is %d generated ROOT files' % len(rootFilesList_0))
#rootFilesList_0 = rootFilesList_0[0:nbFiles]
nbFiles = 1000 #change_nbFiles(len(rootFilesList_0), nbFiles)

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
checkFolder(pathNb_files)
folderNB = pathNb_files
pathKS = pathNb_files + 'KS'
pathKS =checkFolderName(pathKS)
checkFolder(pathKS)
print('pathKS : {:s}'.format(pathKS))

# get list of added ROOT files for comparison
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
print('pathDATA : {:s}'.format(pathDATA))
rootFilesList = getListFiles(pathDATA, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    f_root = ROOT.TFile(pathDATA + item)
    h_rel = gr.getHisto(f_root, tp_1)
    for i in range(0, N_histos): # 1 N_histos histo for debug
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            d = gr.getHistoConfEntry(histo_rel)
            #d = getHistoConfEntry(histo_rel)
            s_tmp = fill_Snew2(d, histo_rel)
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

sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
print('sortedRels[0] = {}'.format(sortedRels[0]))

pathDBox = pathKS + 'DBox'
pathDBox =checkFolderName(pathDBox)
checkFolder(pathDBox)
print('pathDBox : {:s}'.format(pathDBox))

f2 = ROOT.TFile(pathDATA + input_ref_file)
h2 = gr.getHisto(f2, tp_1)

# get histoConfig file
histoArray = []
hCF = pathChiLib + '/HistosConfigFiles/ElectronMcSignalHistos.txt'
fCF = open(hCF, 'r') # histo Config file
for line in fCF:
    if (len(line) != 0):
        if "/" in line:
            histoArray.append(line)

for elem in sortedRels:
    print('=== release : {}'.format(elem))
    rel = elem[1]
    file = elem[2]

    # get the "new" root file datas
    f_rel = ROOT.TFile(pathDATA + file)
    print('we use the {} file as KS relative'.format(file))
    h1 = gr.getHisto(f_rel, tp_1)

    for i in range(0, len(histoArray)): 
        print(histoArray[i]) # print histo name
        
        short_histo_name, short_histo_names, histo_positions = tl.shortHistoName(histoArray[i])
        histo_2 = h2.Get(short_histo_names[0])
        histo_1 = h1.Get(short_histo_names[0])
        gif_name = pathDBox + short_histo_names[0] + '_' + rel + ".gif"
        gr.createPicture2b(histo_1, histo_2, gif_name, 0)
            
    ROOT.TFile.Close(f_rel)

print("Fin !\n")
