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

import os,sys
import importlib
import importlib.machinery
import importlib.util
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

from multiprocessing import Pool

import pandas as pd
import numpy as np
import matplotlib
import itertools

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal # ROOT.kBreak # 
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True

#from ROOT import gROOT
root_version = ROOT.gROOT.GetVersion()

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
    print("step 4 - arg. 3 :", sys.argv[3]) # interactive / batch
    print("step 4 - arg. 4 :", sys.argv[4]) # branche
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
    mode = sys.argv[3]
    branche = sys.argv[4][1:]
else:
    print("rien")
    pathBase = ''
    mode = "b"
    branche =''

sys.path.append(pathCommonFiles)
matplotlib.use('agg')

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\ncheckDiffTest_v3")
print('branche : {:s}'.format(branche))
print(len(branche))

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

import validationsDefault as dfo
from rootValues import NB_EVTS
from controlFunctions import checkFolderName, getListFiles, checkFolder, colorText
from graphicFunctions import Graphic
from graphicAutoEncoderFunctions import GraphicKS
from DecisionBox import DecisionBox
from filesSources import input_ref_file

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

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
rootFilesList_0 = rootFilesList_0[0:dfo.nbFiles]
nbFiles = 1000 # change_nbFiles(len(rootFilesList_0), nbFiles)

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
    #print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    name = os.path.join(pathDATA, item)
    f_root = ROOT.TFile(name)
    h_rel = gr.getHisto(f_root, dfo.tp_1)
    histo_rel = h_rel.Get(branche)
    if (histo_rel):
        d = gr.getHistoConfEntry(histo_rel)
        s_tmp = gr.fill_Snew2(d, histo_rel)
        if (s_tmp.min() < 0.):
            print('pbm whith histo %s, min < 0' % branche)
        elif (np.floor(s_tmp.sum()) == 0.):
            print('pbm whith histo %s, sum = 0' % branche)
    else:
        print('%s KO' % branche)

sortedRels = rels # sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

# get the "reference" root file datas
input_ref_file = pathDATA + sortedRels[0][2]
print('input_ref_file : {:s}'.format(input_ref_file))

f_KSref = ROOT.TFile(input_ref_file) # pathDATA + 
print('we use the %s file as KS reference' % input_ref_file)
h_KSref = gr.getHisto(f_KSref, dfo.tp_1)

nbRels = len(sortedRels)
print('histo : {:s}'.format(branche)) # print histo name

s_KSref = []
histo_KSref = h_KSref.Get(branche)
if (histo_KSref):
    print('%s OK' % branche)
    s_KSref = []
    for entry in histo_KSref:
        s_KSref.append(entry)
    s_KSref = np.asarray(s_KSref)
    s_KSref = s_KSref[1:-1]
print('')

tic = time.time()

if (mode == "i"):
    print('[histo : {:s}]'.format(colorText(branche, 'green'))) # print histo name interactif
else:
    print('[histo : {:s}]'.format(branche)) # print histo name batch

# get nb of columns & rows for histos
Ncols = len(s_KSref)
print('Ncols : [%d]' % (Ncols))

t_leaf = [] # get the max for each ROOT file

for item in rootFilesList_0:
    cc = (item.split('.')[0]).split('_')[-1]
    if (int(cc) < nbFiles):
        name = pathROOTFiles + item
        f_Rootf = ROOT.TFile(name)
        h_Rootf = gr.getHisto(f_Rootf, dfo.tp_1)
        histo_Rootf = h_Rootf.Get(branche)
        s_Rootf = []
        for entry in histo_Rootf:
            s_Rootf.append(entry)
        s_Rootf = np.asarray(s_Rootf)
        s_Rootf = s_Rootf[1:-1]
    
        s0 = np.asarray(s_KSref) # if not this, ind is returned as b_00x instead of int value
        s1 = np.asarray(s_Rootf)
        N0 = len(s0)
        N1 = len(s1)
        if (N0 != N1):
            print('not the same lengths')
            print('s0 has {:d} elements'.format(N0))
            print('s1 has {:d} elements'.format(N1))
            exit()
        min0 = s0.min() # min(s0)
        min1 = s1.min() # min(s1)
        min01 = np.min([min0, min1])
        if (min01 > 0.):
            min01 = 0.
        else:
            min01 = np.abs(min01)
        SumSeries0 = s0.sum() + N0 * min01
        SumSeries1 = s1.sum() + N1 * min01
        v0, v1 = 0., 0.
        sDKS = []
        s0 = (s0 + min01) / SumSeries0
        s1 = (s1 + min01) / SumSeries1
        for j in range(0, N0):
            v0 += s0[j]
            v1 += s1[j]
            sDKS.append(v1 - v0)
        sDKS = np.abs(sDKS)
        t_leaf.append(np.max(sDKS))
        f_Rootf.Close()
t_leaf2 = [] # get the min/max of t_leaf
t_leaf2.append(np.min(t_leaf))
t_leaf2.append(np.max(t_leaf))
print('plage : ', t_leaf2)

# draw the picture with KS plot and diff position
#fileName1 = pathKS + '/KS-ttlDiff_1_' + branche + '_v6.png'
#legende = [ 'ROOT vs ref diff', 'min/max [ {:.3e} + ,  + {:.3e} + ]'.format(t_leaf2[0], t_leaf2[1]) ]
#grKS.createSimpleDiffPicture2(branche + ' : ' + str(N0), t_leaf, t_leaf2, ['bins', 'norm. diff.'], legende, fileName1)

name = pathROOTFiles + "histo_" + branche + '_{:03d}'.format(nbFiles) + ".txt"
df = pd.read_csv(name)

# check the values data
cols = df.columns.values
cols_entries = cols[7::2]
df_entries = df[cols_entries]
df_GetEntries = df['nbBins'] # nbBins (GetEntries())

# get nb of columns & rows for histos
(Nrows, Ncols) = df_entries.shape
print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
df_entries = df_entries.iloc[:, 1:Ncols-1]
(Nrows, Ncols) = df_entries.shape
print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))

totalDiff = []

# INNER MULTIPROCESSING
def getSeriesDiff(k_lj):
    k, lj, df_entries = k_lj
    series0 = df_entries.iloc[k, :]
    series1 = df_entries.iloc[lj, :]
    return DB.diffMAXKS3b(series0, series1)
with Pool() as pool:
    args = [(k,lj, df_entries) for k, lj in itertools.combinations(range(Nrows), 2)]
    totalDiff = pool.map(getSeriesDiff, args)

# Kolmogoroff-Smirnov curve
seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=dfo.nbins)

# draw the picture with KS plot and diff position
fileName1 = pathKS + '/KS-ttlDiff_1_' + branche + '_v7b.png'
grKS.createSimpleKSttlDiffPicture2(totalDiff, dfo.nbins, branche + ' : ' + str(Ncols), fileName1, s_KSref, t_leaf2)
print(' ')

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !\n")

