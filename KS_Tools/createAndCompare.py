#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# createAndCompare : create file for Kolmogorov-Smirnov maximum diff and generate
# pictures for releases comparison
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

from genericpath import exists
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

from ROOT import gROOT
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

import pandas as pd
import numpy as np
import matplotlib

print('PANDAS     version : {}'.format(pd.__version__))
print('PYTHON     version : {}'.format(sys.version))
print("NUMPY      version : {}".format(np.__version__))
print('MATPLOTLIB version : {}'.format(matplotlib.__version__))
print("ROOT      version : {}".format(root_version))

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\ncreateAndCompare")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, pathCommonFiles+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
pathBase = blo.RESULTFOLDER 
print('result path : {:s}'.format(pathBase))

pathChiLib = pathLIBS + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(pathChiLib))
sys.path.append(pathChiLib)
sys.path.append(pathCommonFiles)

import validationsDefault as dfo
from validationsDefault import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import Graphic
from graphicAutoEncoderFunctions import GraphicKS, createCompLossesPicture
from DecisionBox import DecisionBox
from filesSources import *
from rootSources import *

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
'''rootFilesList_0 = getListFiles(pathROOTFiles) # get the list of the root files in the folderName folder
if (len(rootFilesList_0) ==0 ):
    print('there is no generated ROOT files')
    exit()
rootFilesList_0.sort()
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
rootFilesList_0 = rootFilesList_0[0:nbFiles]'''
nbFiles = 1000

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
print('pathNb_files après check : %s' % pathNb_files)
checkFolder(pathNb_files)
pathKS = pathNb_files + 'KS'
pathKS =checkFolderName(pathKS)
print('folder KS après check : %s' % pathKS)
checkFolder(pathKS)

# get list of added ROOT files for comparison
pathDATA = pathLIBS + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'

tmpSource1 = []
for elem in rootSources:
    if ((elem[0] == '+') or (elem[0] == '*')):
        tmpSource1.append(elem[1])
        if (elem[0] == '*'):
            # extract release from source reference
            input_ref_file = elem[1]
            release = elem[1].split('__')[2].split('-')[0]
            print('extracted release : {:s}'.format(release))
rootFilesList = []
for elem in tmpSource1:
    name = pathDATA + '/' + elem
    if exists(name):
        rootFilesList.append(elem)
print(rootFilesList)

print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    f_root = ROOT.TFile(pathDATA + item)
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

source_dest = pathNb_files + "/ElectronMcSignalHistos.txt"
shutil.copy2(source, source_dest)

sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
# get the "reference" root file datas
f_KSref = ROOT.TFile(pathDATA + input_ref_file)
print('we use the %s file as KS reference' % input_ref_file)

h_KSref = gr.getHisto(f_KSref, tp_1)
print(h_KSref)

for elem in sortedRels:
    rel = elem[1]
    
    KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
    print("KSname 1 : %s" % KS_diffName)
    wKS0 = open(KS_diffName, 'w')
    wKS0.close()

    KS_diffName_std = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-std.txt"
    print("KSname 1 : %s" % KS_diffName_std)
    wKS4 = open(KS_diffName_std, 'w')
    wKS4.close()

    KS_diffName_mean = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-mean.txt"
    print("KSname 1 : %s" % KS_diffName_mean)
    wKS5 = open(KS_diffName_mean, 'w')
    wKS5.close()


nbRels = len(sortedRels)

tic = time.time()

for i in range(0, N_histos):#, N_histos-1 range(N_histos - 1, N_histos):  # 1 N_histos histo for debug
    print('histo : {:s}'.format(branches[i])) # print histo name
    
    histo_rel = h_rel.Get(branches[i])
    if (histo_rel):
        print('%s OK' % branches[i])
        name = pathROOTFiles + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
        print('\n%d - %s' %(i, name))
        '''df = pd.read_csv(name)
    
        # check the values data
        cols = df.columns.values
        n_cols = len(cols)
        print('nb of columns for histos : %d' % n_cols)
        cols_entries = cols[6::2]
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
        nb1 = 0
        totalDiff = []
        for k in range(0,Nrows-1):
            for l in range(k+1, Nrows):
                nb1 += 1
                series0 = df_entries.iloc[k,:]
                series1 = df_entries.iloc[l,:]     
                sum0 = df_GetEntries[k]
                sum1 = df_GetEntries[l]
                totalDiff.append(DB.diffMAXKS3(series0, series1)[0]) # 9000, 9000

        print('ttl nb of couples 1 : %d' % nb1)'''

        # create the datas for the p-Value graph
        # by comparing 1 curve with the others.
        histo_KSref = h_KSref.Get(branches[i])
        s_KSref = []
        for entry in histo_KSref:
            s_KSref.append(entry)
        s_KSref = np.asarray(s_KSref)
        s_KSref = s_KSref[1:-1]

        Ntot_h_KSref = histo_KSref.GetEntries()

        print('\nWorking with sorted rels\n')
        ind_rel = 0
        for elem in sortedRels:
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
                
            # diff max between new & old
            diffMax0, posMax0, sDKS = DB.diffMAXKS3(s_KSref, s_new)
            print("diffMax0 : %f - posMax0 : %f" % (diffMax0, posMax0))
            print('ind rel : {:d} : {:s} : {:e}\n'.format(ind_rel, branches[i], diffMax0)) # OK
            KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
            wKS0 = open(KS_diffName, 'a')
            wKS0.write('{:s} : {:e}\n'.format(branches[i], diffMax0))
            wKS0.close()
            diffMax4, posMax4, sDKS = DB.diffMAXKS4(s_KSref, s_new)
            print("diffMax4 : %f - posMax4 : %f" % (diffMax4, posMax4))
            KS_diffName_std = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-std.txt"
            wKS4 = open(KS_diffName_std, 'a')
            wKS4.write('{:s} : {:e}\n'.format(branches[i], diffMax4))
            wKS4.close()
            diffMax5, posMax5, sDKS = DB.diffMAXKS5(s_KSref, s_new)
            print("diffMax5 : %f - posMax5 : %f" % (diffMax5, posMax5))
            KS_diffName_mean = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-mean.txt"
            wKS5 = open(KS_diffName_mean, 'a')
            wKS5.write('{:s} : {:e}\n'.format(branches[i], diffMax5))
            wKS5.close()

            # print nb of red/green lines
            print('[ind_rel/nbRels] : [{:d}/{:d}]'.format(ind_rel, nbRels))
            ind_rel += 1
    else:
        print('%s KO' % branches[i])

# generate pictures
df_ttl0 = []
df_ttl4 = []
df_ttl5 = []

for elem in sortedRels:
    print(elem)
    rel = elem[1]

    # get the KS file datas
    KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
    KS_diffName_std = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-std.txt"
    KS_diffName_mean = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + "-mean.txt"
    if exists(KS_diffName):
        print('%s existe'%KS_diffName)
    else:
        print('%s n\'existe pas'%KS_diffName)
    if exists(KS_diffName_std):
        print('%s existe'%KS_diffName_std)
    else:
        print('%s n\'existe pas'%KS_diffName_std)
    if exists(KS_diffName_mean):
        print('%s existe'%KS_diffName_mean)
    else:
        print('%s n\'existe pas'%KS_diffName_mean)

    wKS0 = open(KS_diffName, 'r').readlines()
    wKS4 = open(KS_diffName_std, 'r').readlines()
    wKS5 = open(KS_diffName_mean, 'r').readlines()
    sum0 = 0.
    sum4 = 0.
    sum5 = 0.

    print(len(wKS0))
    print(len(wKS4))
    print(len(wKS5))
    
    # diff
    tmpArr1 = []
    tmpArr2 = []
    
    nbLines = 0
    for line in wKS0:
        #print(len(line))
        aa = line.split(' : ')
        print(aa[0])
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1][:-1]))
        sum0 += float(aa[1][:-1])
        #print('{:s} : {:f}'.format(aa[0], sum0))
        nbLines += 1
    df_ttl0.append([rel, sum0/nbLines])

    tmpArr1 = []
    tmpArr2 = []
    nbLines = 0
    for line in wKS4:
        #print(len(line))
        aa = line.split(' : ')
        print(aa[0])
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1][:-1]))
        sum4 += float(aa[1][:-1])
        nbLines += 1
    df_ttl4.append([rel, sum4/nbLines])

    tmpArr1 = []
    tmpArr2 = []
    nbLines = 0
    for line in wKS5:
        #print(len(line))
        aa = line.split(' : ')
        print(aa[0])
        tmpArr1.append(aa[0])
        tmpArr2.append(float(aa[1][:-1]))
        sum5 += float(aa[1][:-1])
        nbLines += 1
    df_ttl5.append([rel, sum5/nbLines])


print(df_ttl0)
print()
print(df_ttl4)
print()
print(df_ttl5)
print()

# histo complet recapitulatif
lab = []
val = []
for elem in df_ttl0:
    lab.append(elem[0])
    val.append(elem[1])
#print(val)
pictureName = pathKS + 'comparison_KS_values_total_cum_{:03d}'.format(nbFiles) +'.png' # 
print(pictureName)
title = r"$\bf{total}$" + ' : KS cum diff values vs releases.'
createCompLossesPicture(lab,val, pictureName, title)

lab = []
val = []
for elem in df_ttl4:
    lab.append(elem[0])
    val.append(elem[1])
#print(val)
pictureName = pathKS + 'comparison_KS_values_total_std_{:03d}'.format(nbFiles) +'.png' # 
print(pictureName)
title = r"$\bf{total}$" + ' : KS std diff values vs releases.'
createCompLossesPicture(lab,val, pictureName, title)

lab = []
val = []
for elem in df_ttl5:
    lab.append(elem[0])
    val.append(elem[1])
print(val)
pictureName = pathKS + 'comparison_KS_values_total_mean_{:03d}'.format(nbFiles) +'.png' # 
#print(pictureName)
title = r"$\bf{total}$" + ' : KS mean diff values vs releases.'
createCompLossesPicture(lab,val, pictureName, title)

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !")
