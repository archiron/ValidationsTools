#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# extractNewFilesValues: create file with values for each histo 
# for different releases of the added ROOT files
# for egamma validation comparison                              
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
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True

#from ROOT import *
root_version = ROOT.gROOT.GetVersion()

print('PYTHON     version : {}'.format(sys.version))
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

print("\nextractNewFilesValues")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('pathChiLib : {:s}'.format(pathChiLib))
print('pathCommonFiles : {:s}'.format(pathCommonFiles))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import validationsDefault as dfo
from validationsDefault import *
from rootValues import NB_EVTS
from controlFunctions import getBranches, cleanBranches, checkFolderName, getListFiles, change_nbFiles, checkFolder
from graphicFunctions import Graphic 
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

print("func_ExtractNewFilesValues")

pathNb_evts = pathBase + '/' + '{:04d}'.format(NB_EVTS) + '/' + release
pathNb_evts = checkFolderName(pathNb_evts)
print('pathNb_evts : {:s}'.format(pathNb_evts))
pathROOTFiles = blo.pathROOT + "/" + release
print('pathROOTFiles : {:s}'.format(pathROOTFiles))

# these line for daltonians !
#seaborn.set_palette('colorblind')

gr = Graphic()
gr.initRoot()

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathROOTFiles)
if (len(rootFilesList_0) ==0 ):
    print('there is no generated ROOT files')
    exit()
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)
pathCase = pathNb_evts + checkFolderName(dfo.folder)
pathNb_files = pathCase + '{:03d}'.format(nbFiles)
print('pathNb_files : {:s}'.format(pathNb_files))
checkFolder(pathNb_files)

# get list of the added ROOT files for comparison
pathDATA = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Run3/RECO/') # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
print('pathDATA : {:s}'.format(pathDATA))
rootFilesList = getListFiles(pathDATA, 'root')
rootFilesList2 = []
rootList2 = os.path.join(pathLIBS, blo.DATA_SOURCE + '/Values/rootSourcesRelValZEE_14mcRun3RECO/rootSourcesRelValZEE_14mcRun3RECO.txt')
print('root liste2 : {:s}'.format(rootList2))

sourceList = open(rootList2, "r")
#compteur = 0
for ligne in sourceList:
    t_ligne = ligne.replace('_0.txt', '.root')
    t_ligne = t_ligne.replace('_1.txt', '.root')
    #print('{:2d} : [{:s}] - [{:s}]'.format(compteur, ligne, t_ligne))
    rootFilesList2.append(t_ligne.rstrip())
    #compteur += 1
compteur = 0
for item in rootFilesList2:
    print('\n{:2d} : {:s}'.format(compteur, item))
    compteur += 1
rootFilesList3 = []
for item in rootFilesList2: 
    if item not in rootFilesList3: 
        rootFilesList3.append(item)
        print(item)
print('Root files List have {:d} files'.format(len(rootFilesList3)))
#Stop()
print('we use the files :')
print('there is ' + '{:03d}'.format(len(rootFilesList3)) + ' added ROOT files')
if (len(rootFilesList3) == 0):
    print('no added ROOT files to work with. Existing.')
    exit()
for item in rootFilesList3:
    print('%s' % item)

N_histos = len(branches)
print('N_histos : %d' % N_histos)
KS_resume = pathNb_files + "/branchesHistos_NewFiles.txt"
print("KSname 0 : %s" % KS_resume)
wKS_ = open(KS_resume, 'w')

tic = time.time()

for i in range(0, N_histos): # 1 histo for debug
    KS_resume2 = pathNb_files + "/branchesHistos_NewFiles_" + branches[i] + '.txt'
    print("KSname 0 : %s" % KS_resume)
    print("KSname 0 : %s" % KS_resume2)
    wKS2_ = open(KS_resume2, 'w')

    for file in rootFilesList3:
        # extract release version
        fil = file.split('__')[2]
        fil = fil.split('-')[0]
        inputFile = pathDATA + file
        rFile = ROOT.TFile(inputFile)
        h1 = gr.getHisto(rFile, tp_1)
        print(fil + '/' + branches[i]) # print histo name
        histo_1 = h1.Get(branches[i])

        if (histo_1):
            print('%s OK' % branches[i])
            d = gr.getHistoConfEntry(histo_1)
            #print("d = {}".format(d))

            texttoWrite = fil + "," + branches[i] + ","
            s_new = gr.fill_Snew2(d, histo_1)
        
            for elem in s_new:
                texttoWrite += str(elem) + ","
            texttoWrite = texttoWrite[:-1] # remove last char
            texttoWrite += '\n'
            wKS_.write(texttoWrite)
            wKS2_.write(texttoWrite)
        else:
            print('%s KO' % branches[i])
    wKS2_.close()

wKS_.close()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

