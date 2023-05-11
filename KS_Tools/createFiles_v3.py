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

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

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

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\ncreateFiles_v3")

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
from graphicAutoEncoderFunctions import GraphicKS
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

DB = DecisionBox()
grKS = GraphicKS()
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

# get list of generated ROOT files
rootFilesList_0 = getListFiles(pathNb_evts, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

pathNb_files = pathCase + '{:03d}'.format(nbFiles)
pathNb_files = checkFolderName(pathNb_files)
checkFolder(pathNb_files)
pathKS = pathNb_files + 'KS'
pathKS =checkFolderName(pathKS)
checkFolder(pathKS)

source_dest = pathNb_files + "/ElectronMcSignalHistos.txt"
shutil.copy2(source, source_dest)

#print('-')
#for elem in rels:
#    print(elem)
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
#print('-')
#LOG_SOURCE_WORK= #'/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles/'
# get the "reference" root file datas
f_KSref = ROOT.TFile(pathDATA + input_ref_file)
print('we use the %s file as KS reference' % input_ref_file)

if (ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)
h_KSref = getHisto(f_KSref, tp_1)
print(h_KSref)

wKS0_Files = []
wKS__Files = []
wKSp_Files = []
for elem in sortedRels:
    rel = elem[1]
    
    KS_diffName = pathNb_files + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
    print("KSname 1 : %s" % KS_diffName)
    wKS0 = open(KS_diffName, 'w')
    wKS0_Files.append(wKS0)

    KS_resume = pathNb_files + "/histo_resume" + "_" + rel + ".txt"
    print("KSname 0 : %s" % KS_resume)
    wKS_ = open(KS_resume, 'w')
    wKS__Files.append(wKS_)

    KS_pValues = pathNb_files + "/histo_pValues" + "_" + rel + ".txt"
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'w')
    wKSp_Files.append(wKSp)

nbRels = len(sortedRels)
redGreen1 = [[0 for c in range(2)] for r in range(nbRels)]
redGreen2 = [[0 for c in range(2)] for r in range(nbRels)]
redGreen3 = [[0 for c in range(2)] for r in range(nbRels)]
'''
print('{:d}/2 '.format(nbRels))
for i in range(0, nbRels):
    for j in range(0, 2):
        print('{:d}/{:d} : {:d}'.format(i,j,redGreen1[i][j]))
for i in range(0, nbRels):
    for j in range(0, 2):
        print('{:d}/{:d} : {:d}'.format(i,j,redGreen2[i][j]))
'''
tic = time.time()

for i in range(0, N_histos): # 1 N_histos histo for debug
    print('histo : {:s}'.format(branches[i])) # print histo name
    
    histo_rel = h_rel.Get(branches[i])
    if (histo_rel):
        print('%s OK' % branches[i])
        name = pathNb_evts + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
        print('\n%d - %s' %(i, name))
        df = pd.read_csv(name)
    
        # check the values data
        #print(df.head(5))
        cols = df.columns.values
        n_cols = len(cols)
        print('nb of columns for histos : %d' % n_cols)
        cols_entries = cols[6::2]
        df_entries = df[cols_entries]
        #print(df_entries.head(15))#

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
                #totalDiff.append(DB.diffMAXKS3(series0, series1, sum0, sum1)[0]) # 9000, 9000
                totalDiff.append(DB.diffMAXKS3(series0, series1)[0]) # 9000, 9000

        print('ttl nb1 of couples 1 : %d' % nb1)

        # create the datas for the p-Value graph
        # by comparing 1 curve with the others.
        # Get a random histo as reference (KS 2)
        series_reference = df_entries.iloc[ind_reference,:]
        nbBins_reference = df_GetEntries[ind_reference]
        print('nb bins reference : %d' % nbBins_reference)
        #print(series_reference)
        nb2 = 0
        totalDiff2 = []
        for k in range(0,Nrows-0):
            if (k != ind_reference):
                nb2 += 1
                series0 = df_entries.iloc[k,:]
                sum0 = df_GetEntries[k]
                #totalDiff2.append(DB.diffMAXKS3(series0, series_reference, sum0, nbBins_reference)[0]) # 9000, 9000
                totalDiff2.append(DB.diffMAXKS3(series0, series_reference)[0]) # 9000, 9000

        print('ttl nb of couples 2 : %d' % nb2)
        
        histo_KSref = h_KSref.Get(branches[i])
        #ii=0
        s_KSref = []
        for entry in histo_KSref:
            #print("%d/%d : %s - %s") % (ii, histo_KSref.GetXaxis().GetNbins(), entry, histo_KSref.GetBinError(i))
            s_KSref.append(entry)
            #ii += 1
        s_KSref = np.asarray(s_KSref)
        s_KSref = s_KSref[1:-1]
        #print(s_KSref)
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

            nb_red1 = 0
            nb_green1 = 0
            nb_red2 = 0
            nb_green2 = 0
            nb_red3 = 0
            nb_green3 = 0

            h_rel = getHisto(f_rel, tp_1)
            histo_rel = h_rel.Get(branches[i])

            #ii=0
            s_new = []
            for entry in histo_rel:
                #print("%d/%d : %s - %s") % (ii, histo_rel.GetXaxis().GetNbins(), entry, histo_rel.GetBinError(i))
                s_new.append(entry)
                #ii += 1
            s_new = np.asarray(s_new)
            s_new = s_new[1:-1]
            #print(s_new)
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
            KSname2 = pathNb_files + "/histo_" + branches[i] + "_KScurve2" + "_" + rel + ".txt"
            KSname3 = pathNb_files + "/histo_" + branches[i] + "_KScurve3" + "_" + rel + ".txt"
            print("KSname 1 : %s" % KSname1)
            print("KSname 2 : %s" % KSname2)
            print("KSname 3 : %s" % KSname3)
            wKS1 = open(KSname1, 'w')
            wKS2 = open(KSname2, 'w')
            wKS3 = open(KSname3, 'w')

            # create the datas for the p-Value graph
            # by comparing the new curve with the others.
            # Get the new as reference (KS 3)
            nb3 = 0
            totalDiff3 = []
            for k in range(0,Nrows-0):
                nb3 += 1
                series0 = df_entries.iloc[k,:]
                sum0 = df_GetEntries[k]
                #totalDiff3.append(DB.diffMAXKS3(series0, s_new, sum0, Ntot_h_rel)[0])
                totalDiff3.append(DB.diffMAXKS3(series0, s_new)[0])

            print('ttl nb of couples 3 : %d' % nb3)
        
            # plot some datas
            plt_entries = df_entries.plot(kind='line')
            fig = plt_entries.get_figure()
            fig.clf()
            # create the integrated curve
            curves = []
            for k in range(0,Nrows):
                series0 = df_entries.iloc[k,:]
                curves = DB.funcKS(series0)
                plt.plot(curves)
            fig.savefig(pathKS + '/cumulative_curve_' + branches[i] + "_" + rel + '.png')
            fig.clf()
        
            # ================================ #
            # create the mean curve of entries #
            # ================================ #
            mean_df_entries = df_entries.mean()
            mean_sum = mean_df_entries.sum()
                
            #diffMax1, posMax1 = DB.diffMAXKS3(mean_df_entries, s_new, mean_sum, Ntot_h_rel)
            diffMax1, posMax1, _ = DB.diffMAXKS3(mean_df_entries, s_new)
            #diffMax2, posMax2 = DB.diffMAXKS3(series_reference, s_new, nbBins_reference, Ntot_h_rel)
            diffMax2, posMax2, _ = DB.diffMAXKS3(series_reference, s_new)
            #diffMax3, posMax3 = DB.diffMAXKS3(s_new, s_KSref, Ntot_h_rel, Ntot_h_KSref)
            diffMax3, posMax3, _ = DB.diffMAXKS3(s_new, s_KSref)
            print("diffMax1 : %f - posMax1 : %f" % (diffMax1, posMax1))
            print("diffMax2 : %f - posMax2 : %f" % (diffMax2, posMax2))
            print("diffMax3 : %f - posMax3 : %f" % (diffMax3, posMax3))
            print('Ntot_h_rel : %d - Ntot_h_KSref : %d' % (Ntot_h_rel, Ntot_h_KSref))

            # diff max between new & old
            #diffMax0, posMax0, sDKS = DB.diffMAXKS3(s_KSref, s_new, Ntot_h_KSref, Ntot_h_rel)
            diffMax0, posMax0, sDKS = DB.diffMAXKS3(s_KSref, s_new)
            print("diffMax0 : %f - posMax0 : %f" % (diffMax0, posMax0))
            wKS0_Files[ind_rel].write('%s : %e\n' % (branches[i], diffMax0))
            print(s_new[0:8])
            print(s_KSref[0:8])
            print(sDKS[0:8]) # diff

            yellowCurve1 = mean_df_entries
            yellowCurve2 = series_reference
            yellowCurve3 = s_new
            yellowCurveCum1 = DB.funcKS(mean_df_entries) #  cumulative yellow curve
            yellowCurveCum2 = DB.funcKS(series_reference)
            yellowCurveCum3 = DB.funcKS(s_new)

            # Kolmogoroff-Smirnov curve
            seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
            KSDiffname1 = pathNb_files + '/KSDiffValues_1_' + branches[i] + "_" + rel+ '.txt' # csv imposed by pd.to_csv + "_" + rel 
            df.to_csv(KSDiffname1)
            print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.min()))
            print('\ndiffMax0/sTD.max 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.max()))

            fileName1 = pathKS + '/KS-ttlDiff_1_' + branches[i] + "_" + rel + '.png'
            [nb_green1, nb_red1] = grKS.createKSttlDiffPicture(totalDiff, nbins, diffMax0,'KS diff. 1', fileName1)

            count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            KSDiffHistoname1 = pathNb_files + '/KSDiffHistoValues_1_' + branches[i] + "_" + rel + '.txt'
            wKSDiff1 = open(KSDiffHistoname1, 'w')
            wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in count))
            wKSDiff1.write('\n')
            wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in division))
            wKSDiff1.write('\n')
            wKSDiff1.close()

            # Get the max of the integral
            I_max = DB.integralpValue(division, count, 0.)
            # print the min/max values of differences
            pValue = DB.integralpValue(division, count, diffMax0)
            # save the KS curves
            wKS1.write('%e, %d\n' % (I_max, nbins))
            wKS1.write('%e, %e\n' % (div_min, div_max))
            wKS1.write(' '.join("{:10.04e}".format(x) for x in count))
            wKS1.write('\n')
            wKS1.write(' '.join("{:10.04e}".format(x) for x in division))
            wKS1.write('\n')
            wKS1.write(' '.join("{:10.04e}".format(x) for x in yellowCurve1 )) # average (mean) curve
            wKS1.write('\n')
            wKS1.write(' '.join("{:10.04e}".format(x) for x in yellowCurveCum1 ))
            wKS1.write('\n')
            wKS1.close()

            # Kolmogoroff-Smirnov curve 2
            seriesTotalDiff2 = pd.DataFrame(totalDiff2, columns=['KSDiff'])
            print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff2.values.min()))
            print('\ndiffMax0/sTD.max 2 : %f/%f' % (diffMax0, seriesTotalDiff2.values.max()))

            fileName2 = pathKS + '/KS-ttlDiff_2_' + branches[i] + "_" + rel + '.png'
            [nb_green2, nb_red2] = grKS.createKSttlDiffPicture(totalDiff2, nbins, diffMax0,'KS diff. 2', fileName2)

            count, division = np.histogram(seriesTotalDiff2, bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            KSDiffHistoname2 = pathNb_files + '/KSDiffHistoValues_2_' + branches[i] + "_" + rel + '.txt'
            wKSDiff2 = open(KSDiffHistoname2, 'w')
            wKSDiff2.write(' '.join("{:10.04e}".format(x) for x in count))
            wKSDiff2.write('\n')
            wKSDiff2.write(' '.join("{:10.04e}".format(x) for x in division))
            wKSDiff2.write('\n')
            wKSDiff2.close()
        
            # Get the max of the integral
            I_max2 = DB.integralpValue(division, count, 0.)
            ##print('\nMax. integral 2 : %0.4e for nbins=%d' % (I_max2, nbins))
            # print the min/max values of differences
            ##print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x2))
            pValue2 = DB.integralpValue(division, count, diffMax0)
            #print(division)
            #print(count)
            # save the KS curves
            wKS2.write('%e, %d\n' % (I_max2, nbins))
            wKS2.write('%e, %e\n' % (div_min, div_max))
            wKS2.write(' '.join("{:10.04e}".format(x) for x in count))
            wKS2.write('\n')
            wKS2.write(' '.join("{:10.04e}".format(x) for x in division))
            wKS2.write('\n')
            wKS2.write(' '.join("{:10.04e}".format(x) for x in yellowCurve2 )) # random curve
            wKS2.write('\n')
            wKS2.write(' '.join("{:10.04e}".format(x) for x in yellowCurveCum2 ))
            wKS2.write('\n')
            wKS2.close()

            # Kolmogoroff-Smirnov curve 3
            seriesTotalDiff3 = pd.DataFrame(totalDiff3, columns=['new'])
            print('\ndiffMin0/sTD.min 3 : %f/%f' % (diffMax0, seriesTotalDiff3.values.min()))
            print('diffMax0/sTD.max 3 : %f/%f' % (diffMax0, seriesTotalDiff3.values.max()))
            
            fileName3 = pathKS + '/KS-ttlDiff_3_' + branches[i] + "_" + rel + '.png'
            [nb_green3, nb_red3] = grKS.createKSttlDiffPicture(totalDiff3, nbins, diffMax0,'KS diff. 3', fileName3)
            
            count, division = np.histogram(seriesTotalDiff3, bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            KSDiffHistoname3 = pathNb_files + '/KSDiffHistoValues_3_' + branches[i] + "_" + rel + '.txt'
            wKSDiff3 = open(KSDiffHistoname3, 'w')
            wKSDiff3.write(' '.join("{:10.04e}".format(x) for x in count))
            wKSDiff3.write('\n')
            wKSDiff3.write(' '.join("{:10.04e}".format(x) for x in division))
            wKSDiff3.write('\n')
            wKSDiff3.close()
        
            # Get the max of the integral
            I_max3 = DB.integralpValue(division, count, 0.)
            ##print('\nMax. integral 3 : %0.4e for nbins=%d' % (I_max3, nbins))
            # print the min/max values of differences
            ##print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x3))
            pValue3 = DB.integralpValue(division, count, diffMax0)
            #print(division)
            #print(count)
            # print the p-Value
            print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue, nbins))
            print('normalized p_Value : %0.4e for nbins=%d' % (pValue/I_max, nbins))
            # print the p-Value 2
            print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue2, nbins))
            print('normalized p_Value : %0.4e for nbins=%d' % (pValue2/I_max2, nbins))
            # print the p-Value 3
            print('\nunormalized p_Value : %0.4e for nbins=%d' % (pValue3, nbins))
            print('normalized p_Value : %0.4e for nbins=%d' % (pValue3/I_max3, nbins))

            wKSp_Files[ind_rel].write('%s, %e, %e, %e\n' % (branches[i], pValue/I_max, pValue2/I_max2, pValue3/I_max3))

            plt.close('all')
            # save the KS curves
            wKS3.write('%e, %d\n' % (I_max3, nbins))
            wKS3.write('%e, %e\n' % (div_min, div_max))
            wKS3.write(' '.join("{:10.04e}".format(x) for x in count))
            wKS3.write('\n')
            wKS3.write(' '.join("{:10.04e}".format(x) for x in division))
            wKS3.write('\n')
            wKS3.write(' '.join("{:10.04e}".format(x) for x in yellowCurve3 )) # new curve
            wKS3.write('\n')
            wKS3.write(' '.join("{:10.04e}".format(x) for x in yellowCurveCum3 ))
            wKS3.write('\n')
            wKS3.close()

            # print nb of red/green lines
            print('KS 1 : %d red - %d green for %s' % (nb_red1, nb_green1, branches[i]))
            print('KS 2 : %d red - %d green for %s' % (nb_red2, nb_green2, branches[i]))
            print('KS 3 : %d red - %d green for %s' % (nb_red3, nb_green3, branches[i]))
            nb_red = nb_red1 + nb_red2 + nb_red3
            nb_green = nb_green1 + nb_green2 + nb_green3
            print('KS ttl : %d red - %d green for %s' % (nb_red, nb_green, branches[i]))
            print('[ind_rel/nbRels] : [{:d}/{:d}]'.format(ind_rel, nbRels))
            wKS__Files[ind_rel].write('KS 1 : %d red - %d green for %s\n' % (nb_red1, nb_green1, branches[i]))
            wKS__Files[ind_rel].write('KS 2 : %d red - %d green for %s\n' % (nb_red2, nb_green2, branches[i]))
            wKS__Files[ind_rel].write('KS 3 : %d red - %d green for %s\n' % (nb_red3, nb_green3, branches[i]))
            wKS__Files[ind_rel].write('KS ttl : %d red - %d green for %s\n' % (nb_red, nb_green, branches[i]))
            redGreen1[ind_rel][0] += nb_red1
            redGreen1[ind_rel][1] += nb_green1
            redGreen2[ind_rel][0] += nb_red2
            redGreen2[ind_rel][1] += nb_green2
            redGreen3[ind_rel][0] += nb_red3
            redGreen3[ind_rel][1] += nb_green3
            ind_rel += 1
    else:
        print('%s KO' % branches[i])

ind_rel = 0
for elem in sortedRels:
    wKS__Files[ind_rel].write('KS 1 : %d ttl red - %d ttl green\n' % (redGreen1[ind_rel][0], redGreen1[ind_rel][1]))
    wKS__Files[ind_rel].write('KS 2 : %d ttl red - %d ttl green\n' % (redGreen2[ind_rel][0], redGreen2[ind_rel][1]))
    wKS__Files[ind_rel].write('KS 3 : %d ttl red - %d ttl green\n' % (redGreen3[ind_rel][0], redGreen3[ind_rel][1]))
    ind_rel += 1

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !\n")

