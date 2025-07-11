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

from multiprocessing import Pool

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
import itertools
#import numba # TEST NUMBA => PAS GLOP

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
    print("step 4 - arg. 3 :", sys.argv[2]) # interactive / batch
    pathCommonFiles = sys.argv[1]
    filePaths = sys.argv[2]
    pathLIBS = sys.argv[1][:-12]
    mode = sys.argv[3]
else:
    print("rien")
    pathBase = ''
    mode = "b"

# import matplotlib.dates as md
matplotlib.use('agg')
#from matplotlib import pyplot as plt

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

tic = time.time()

# extract release from source reference
release = input_ref_file.split('__')[2].split('-')[0]
print('extracted release : {:s}'.format(release))

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
nbFiles = 1000 # change_nbFiles(len(rootFilesList_0), nbFiles)
tac = time.time()

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
    #print('\n{:2d} : {:s}'.format(compteur, item))
    compteur += 1
rootFilesList3 = []
for item in rootFilesList2: 
    if item not in rootFilesList3: 
        rootFilesList3.append(item) 
#compteur = 0
#for item in rootFilesList3:
    #print('\n{:2d} : {:s}'.format(compteur, item))
    #compteur += 1
tuc = time.time()

print('we use the files :')
for item in rootFilesList3:
    tmp_branch = []
    nbHistos = 0
    #print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:], item])
    name = os.path.join(pathDATA, item)
    #print('{:s} : {:d}'.format(item, len(item)))
    #print('name : {:s}'.format(name))
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
#print(branches2)
branches = branches2
tec = time.time()

print('pathNb_files : {:s}'.format(pathNb_files))
source_dest = pathNb_files + "/ElectronMcSignalHistos.txt"
#print('source : {:s}'.format(source))
#print('source_dest : {:s}'.format(source_dest))
shutil.copy2(source, source_dest)

sortedRels = rels # sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted

# get the "reference" root file datas
if (dfo.ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
else:
    ind_reference = dfo.ind_reference

nbRels = len(sortedRels)

N_histos = 10
'''t_histos = [] # not used
t1 = time.time()
for i in range(0, N_histos): # load all histos datas
    print('histo : {:s}'.format(branches[i])) # print histo name
    
    histo_KSref = h_KSref.Get(branches[i])
    if (histo_KSref):
        print('%s OK' % branches[i])
        s_KSref = []
        for entry in histo_KSref:
            s_KSref.append(entry)
        s_KSref = np.asarray(s_KSref)
        t_histos.append(s_KSref[1:-1])
t2 = time.time()'''
print('')

t_time = []
for i in range(0, N_histos):#,N_histos, N_histos-1 range(N_histos - 1, N_histos):  # 1 N_histos histo for debug
    if (mode == "i"):
        print('[{:03s}] : histo : {:s}'.format(colorText(str(i), 'blue'), colorText(branches[i], 'green'))) # print histo name interactif
    else:
        print('[{:03d}] : histo : {:s}'.format(i, branches[i])) # print histo name batch
    
    print('%s OK' % branches[i])
    name = pathROOTFiles + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
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

    # create the datas for the p-Value graph
    # by comparing all curves between them. (KS 1)
    # first try => OK
    t3 = time.time()
    totalDiff = []
    '''for k in range(0,Nrows-1):
        for lj in range(k+1, Nrows):
            series0 = df_entries.iloc[k,:]
            series1 = df_entries.iloc[lj,:]
            #t5 = time.time()
            totalDiff.append(DB.diffMAXKS3b(series0, series1))
            #t6 = time.time()
            #print('[k, l, tps] : [%3d, %3d, %f]' % (k, lj, t6-t5))'''

    # Use combinations to generate unique pairs of row indices => OK
    for k, lj in itertools.combinations(range(Nrows), 2):
        series0 = df_entries.iloc[k, :]
        series1 = df_entries.iloc[lj, :]
        totalDiff.append(DB.diffMAXKS3b(series0, series1))

    # TEST NUMBA => PAS GLOP
    '''@numba.jit(nopython=True)  # Numba optimization
    def process_pairs(Nrows, df_entries):
        results = []
        for k, lj in itertools.combinations(range(Nrows), 2):
            series0 = df_entries.iloc[k, :].values  # Convert to numpy array for faster processing
            series1 = df_entries.iloc[lj, :].values
            N0 = len(series0)
            N1 = len(series1)
            min0 = series0.min() # min(s0)
            min1 = series1.min() # min(s1)
            min01 = min([min0, min1])
            if (min01 > 0.):
                min01 = 0.
            else:
                min01 = np.abs(min01)
            SumSeries0 = series0.sum() + N0 * min01
            SumSeries1 = series1.sum() + N1 * min01
            v0, v1 = 0., 0.
            sDKS = np.empty((0), dtype=float) #[]
            series0 = (series0 + min01) / SumSeries0
            series1 = (series1 + min01) / SumSeries1
            v0, v1 = 0., 0.
            sDKS = [abs(v1 := v1 + series1[i] - (v0 := v0 + series0[i])) for i in range(N0)]
            results.append(0.+np.max(sDKS))
        return results
    Nrows = len(df_entries)  # Number of rows in the DataFrame
    totalDiff = process_pairs(Nrows, df_entries)'''
    
    # TEST MULTIPROCESSING
    '''def getSeriesDiff(k_lj):
        k, lj, df_entries = k_lj
        series0 = df_entries.iloc[k, :]
        series1 = df_entries.iloc[lj, :]
        return DB.diffMAXKS3b(series0, series1)
    with Pool() as pool:
        args = [(k,lj, df_entries) for k, lj in itertools.combinations(range(Nrows), 2)]
        totalDiff = pool.map(getSeriesDiff, args)'''

    t4 = time.time()
    t_time.append(t4-t3)
    #print(totalDiff)
    #print(len(totalDiff))

    # Kolmogoroff-Smirnov curve
    seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
    count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)

    nb1 = len(totalDiff)
    val1 = []
    val2 = []
    moy = []
    m_tot = 0.
    sig_tot = 0.
    if (totalDiff[0] != 0.):
        m_tot += totalDiff[0]
    if (totalDiff[1] != 0.):
        m_tot += totalDiff[1]
    for j in range(2,nb1):
        if(totalDiff[j] != 0.):
            m_tot += totalDiff[j]
        m_moy = m_tot / j # (nb1 - 1)
        if(totalDiff[j] != 0.):
            sig_tot += (totalDiff[j] - m_moy) * (totalDiff[j] - m_moy)
        sig = sig_tot / j # (nb1 - 1)
        sig = np.sqrt(sig)
        #print('[m, sig] : [{:e}, {:e}]'.format(m_moy, sig))
        val1.append(m_moy - sig)
        val2.append(m_moy + sig)
        moy.append(m_moy)

    # draw the picture with KS plot and diff position
    fileName1 = pathKS + '/KS-ttlDiff_1_' + branches[i] + '_v9a.png'
    title = 'stats. ' + branches[i]
    grKS.createStatPicture(val1, val2, moy, fileName1, title, labx='nb', laby='')
    

#Stop()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))
print('time for ROOT files list : {:.4f} seconds'.format(tac-tic))
print('time for DATA files list : {:.4f} seconds'.format(tuc-tac))
print('time for branches : {:.4f} seconds'.format(tec-tuc))
print('time for histos array : {:.4f} seconds'.format(t2-t1))
print('time for complete loop for {:d} histos : {:.4f} seconds'.format(N_histos, toc-t1))
for elem in t_time:
    print('ttl diff : {:.4f}'.format(elem))
t_mean = np.asarray(t_time).mean()
print('mean time for ttl diff : {:.4f}'.format(t_mean))

print("Fin !\n")

