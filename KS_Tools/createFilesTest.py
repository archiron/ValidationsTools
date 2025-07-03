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
print(h_KSref)

for elem in sortedRels:
    rel = elem[1]
    
    KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
    print("KSname 1 : %s" % KS_diffName)
    wKS0 = open(KS_diffName, 'w')
    wKS0.close()

    KS_pValues = pathNb_files + "/histo_pValues" + "_" + rel + ".txt"
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'w')
    wKSp.close()

nbRels = len(sortedRels)
ind_rel = 0

KS_pValues2 = pathNb_files + "/histo_pValues_ttl-v3.txt"
print("KSname 3 : %s" % KS_pValues)
wKSp2 = open(KS_pValues2, 'a')

tic = time.time()

pV_tab = []
for i in range(0, N_histos):#,N_histos, N_histos-1 range(N_histos - 1, N_histos):  # 1 N_histos histo for debug
    print('histo : {:s}'.format(branches[i])) # print histo name
    
    histo_rel = h_rel.Get(branches[i])
    if (histo_rel):
        print('%s OK' % branches[i])
        name = pathROOTFiles + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)
    
        histo_KSref = h_KSref.Get(branches[i])
        s_KSref = []
        for entry in histo_KSref:
            s_KSref.append(entry)
        s_KSref = np.asarray(s_KSref)
        s_KSref = s_KSref[1:-1]

        Ntot_h_KSref = histo_KSref.GetEntries()

        # check the values data
        cols = df.columns.values
        n_cols = len(cols)
        print('nb of columns for histos : %d' % n_cols)
        cols_entries = cols[7::2]
        df_entries = df[cols_entries]

        # nbBins (GetEntries())
        df_GetEntries = df['nbBins']

        # get nb of columns & rows for histos
        (Nrows, Ncols) = df_entries.shape
        print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))
        df_entries = df_entries.iloc[:, 1:Ncols-1]
        (Nrows, Ncols) = df_entries.shape
        print('[Nrows, Ncols] : [%d, %d]' % (Nrows, Ncols))

        # create the datas for the p-Value graph
        # by comparing all curves between them. (KS 1)
        totalDiff = []
        for k in range(0,Nrows-1):
            for lj in range(k+1, Nrows):
                series0 = df_entries.iloc[k,:]
                series1 = df_entries.iloc[lj,:]
                totalDiff.append(DB.diffMAXKS3(series0, series1)[0])

        print('\nWorking with sorted rels\n')
        ind_rel = 0
        for elem in sortedRels: # list of various releases.
            print('[ind_rel/nbRels] : [{:d}/{:d}]'.format(ind_rel, nbRels))
            print(elem)
            rel = elem[1]
            file = elem[2]

            # get the "new" root file datas
            input_rel_file = file
            f_rel = ROOT.TFile(pathDATA + input_rel_file)
            print('we use the %s file as new release' % input_rel_file)

            h_rel = gr.getHisto(f_rel, tp_1)
            histo_rel = h_rel.Get(branches[i])

            s_new = []
            for entry in histo_rel:
                s_new.append(entry)
            s_new = np.asarray(s_new)
            s_new = s_new[1:-1]
            Ntot_h_rel = histo_rel.GetEntries()
            print('Ntot_h_rel : %d - Ntot_h_KSref : %d' % (Ntot_h_rel, Ntot_h_KSref))

            # print min/max for the new curve
            print('\n##########')
            print('min : %f' % s_new.min())
            print('max : %f' % s_new.max())
            print('###########\n')
            if (s_new.min() < 0.):
                print('pbm whith histo %s, min < 0' % branches[i])
                continue
            if (np.floor(s_new.sum()) == 0.):
                print('pbm whith histo %s, sum = 0' % branches[i])
                continue

            # create file for KS curve
            KSname1 = pathNb_files + "/histo_" + branches[i] + "_KScurve1" + "_" + rel + ".txt"
            print("KSname 1 : %s" % KSname1)
            wKS1 = open(KSname1, 'w')
        
            # ================================ #
            # create the mean curve of entries #
            # ================================ #
            mean_df_entries = df_entries.mean()
            mean_sum = mean_df_entries.sum()
                
            diffMax1, posMax1, _ = DB.diffMAXKS3(mean_df_entries, s_new)
            print("diffMax1 : %f - posMax1 : %f" % (diffMax1, posMax1))
            print('Ntot_h_rel : %d - Ntot_h_KSref : %d' % (Ntot_h_rel, Ntot_h_KSref))

            # diff max between new & old
            diffMax0, posMax0, sDKS = DB.diffMAXKS3(s_KSref, s_new)
            print("diffMax0 : %f - posMax0 : %f" % (diffMax0, posMax0))
            print('ind rel : {:d} : {:s} : {:e}\n'.format(ind_rel, branches[i], diffMax0)) # OK
            KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
            wKS0 = open(KS_diffName, 'a')
            wKS0.write('{:s} : {:e}\n'.format(branches[i], diffMax0))
            wKS0.close()
            #print(s_new[0:8])
            #print(s_KSref[0:8])
            #print(sDKS[0:8]) # diff

            # Kolmogoroff-Smirnov curve
            print('Kolmogoroff-Smirnov curve 1')
            seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
            KSDiffname1 = pathNb_files + '/KSDiffValues_1_' + branches[i] + "_" + rel+ '.txt' # csv imposed by pd.to_csv + "_" + rel 
            df.to_csv(KSDiffname1)
            print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.min()))
            print('\ndiffMax0/sTD.max 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.max()))

            count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)
            KSDiffHistoname1 = pathNb_files + '/KSDiffHistoValues_1_' + branches[i] + "_" + rel + '.txt'
            wKSDiff1 = open(KSDiffHistoname1, 'w')
            wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in count))
            wKSDiff1.write('\n')
            wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in division))
            wKSDiff1.write('\n')
            wKSDiff1.close()

            # Get the max of the integral
            I_max = DB.integralpValue(division, count, 0.)
            pValue = DB.integralpValue(division, count, diffMax0)

            # draw the picture with KS plot and diff position
            fileName1 = pathKS + '/KS-ttlDiff_1_' + branches[i] + "_" + rel + '.png'
            grKS.createKSttlDiffPicture2(totalDiff, nbins, diffMax0,'KS diff. 1', fileName1, pValue, I_max)
        
             # print the min/max values of differences
            print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue, nbins))
            print('normalized p_Value : %0.4e for nbins=%d' % (pValue/I_max, nbins))

            KS_pValues = pathNb_files + "/histo_pValues" + "_" + rel + ".txt"
            wKSp = open(KS_pValues, 'a')
            wKSp.write('%s, %e\n' % (branches[i], pValue/I_max))
            wKSp.close()
            pV_tab.append([branches[i], elem[1], pValue/I_max])
            wKSp2.write('{:s}, {:s}, {:e}\n'.format(branches[i], elem[1], pValue/I_max))

            print('[ind_rel/nbRels] : [{:d}/{:d}]'.format(ind_rel, nbRels))
            ind_rel += 1
    else:
        print('%s KO' % branches[i])

#Stop()
wKSp2.close()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !\n")

