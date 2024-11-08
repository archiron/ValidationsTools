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
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nextractFiles_v2")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import validationsDefault as dfo
from validationsDefault import *
from rootValues import NB_EVTS
from controlFunctions import *
from graphicFunctions import getHisto, getHistoConfEntry, fill_Snew2, fill_Snew
from DecisionBox import DecisionBox
from filesSources import *

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))
resultPath = checkFolderName(resultPath)
#folder = checkFolderName(dfo.folder)
folder = resultPath + checkFolderName(dfo.folder)

# get the branches for ElectronMcSignalHistos.txt
######## ===== COMMON LINES ===== ########
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.
######## ===== COMMON LINES ===== ########

DB = DecisionBox()
rels = []
tmp_branches = []
nb_ttl_histos = []

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

# get list of added ROOT files for comparison
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
rootFilesList = getListFiles(rootFolderName, 'root')
print('we use the files :')
for item in rootFilesList:
    tmp_branch = []
    nbHistos = 0
    print('\n%s' % item)
    b = (item.split('__')[2]).split('-')
    #print('%s - %s' % (b[0], b[0][6:]))
    rels.append([b[0], b[0][6:], item])
    f_root = ROOT.TFile(rootFolderName + item)
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
rootFilesList_0 = getListFiles(resultPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder += '{:03d}'.format(nbFiles)
folder = checkFolderName(folder)
checkFolder(folder)
folderKS = folder + 'KS'
folderKS =checkFolderName(folderKS)
checkFolder(folderKS)

source_dest = folder + "/ElectronMcSignalHistos.txt"
shutil.copy2(source, source_dest)

#print('-')
#for elem in rels:
#    print(elem)
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
#print('-')
#LOG_SOURCE_WORK= #'/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles/'
# get the "reference" root file datas
f_KSref = ROOT.TFile(rootFolderName + input_ref_file)
print('we use the %s file as KS reference' % input_ref_file)

if (ind_reference == -1): 
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)
h_KSref = getHisto(f_KSref, tp_1)
print(h_KSref)

tic = time.time()

for elem in sortedRels:
    print(elem)
    rel = elem[1]
    file = elem[2]

    # get the "new" root file datas
    input_rel_file = file
    f_rel = ROOT.TFile(rootFolderName + input_rel_file)
    print('we use the %s file as new release' % input_rel_file)

    nb_red1 = 0
    nb_green1 = 0
    nb_red2 = 0
    nb_green2 = 0
    nb_red3 = 0
    nb_green3 = 0

    KS_diffName = folder + "/histo_differences_KScurve" + "_" + rel + "_" + '_{:03d}'.format(nbFiles) + ".txt"
    print("KSname 1 : %s" % KS_diffName)
    wKS0 = open(KS_diffName, 'w')

    KS_resume = folder + "/histo_resume" + "_" + rel + ".txt"
    print("KSname 0 : %s" % KS_resume)
    wKS_ = open(KS_resume, 'w')

    KS_pValues = folder + "/histo_pValues" + "_" + rel + ".txt"
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'w')

    h_rel = getHisto(f_rel, tp_1)
    for i in range(0, 3): # 1 N_histos histo for debug
        print(branches[i]) # print histo name
        histo_rel = h_rel.Get(branches[i])
        if (histo_rel):
            print('%s OK' % branches[i])
            name = resultPath + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
            print('\n%d - %s' %(i, name))
            df = pd.read_csv(name)
        
            ii=0
            s_new = []
            for entry in histo_rel:
                #print("%d/%d : %s - %s") % (ii, histo_rel.GetXaxis().GetNbins(), entry, histo_rel.GetBinError(i))
                s_new.append(entry)
                ii += 1
            s_new = np.asarray(s_new)
            s_new = s_new[1:-1]
            #print(s_new)
            Ntot_h_rel = histo_rel.GetEntries()

            histo_KSref = h_KSref.Get(branches[i])
            ii=0
            s_KSref = []
            for entry in histo_KSref:
                #print("%d/%d : %s - %s") % (ii, histo_KSref.GetXaxis().GetNbins(), entry, histo_KSref.GetBinError(i))
                s_KSref.append(entry)
                ii += 1
            s_KSref = np.asarray(s_KSref)
            s_KSref = s_KSref[1:-1]
            #print(s_KSref)
            Ntot_h_KSref = histo_KSref.GetEntries()
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
            KSname1 = folder + "/histo_" + branches[i] + "_KScurve1" + "_" + rel + ".txt"
            KSname2 = folder + "/histo_" + branches[i] + "_KScurve2" + "_" + rel + ".txt"
            KSname3 = folder + "/histo_" + branches[i] + "_KScurve3" + "_" + rel + ".txt"
            print("KSname 1 : %s" % KSname1)
            print("KSname 2 : %s" % KSname2)
            print("KSname 3 : %s" % KSname3)
            wKS1 = open(KSname1, 'w')
            wKS2 = open(KSname2, 'w')
            wKS3 = open(KSname3, 'w')

            # check the values data
            #print(df.head(5))
            cols = df.columns.values
            n_cols = len(cols)
            print('nb of columns for histos : %d' % n_cols)
            cols_entries = cols[6::2]
            df_entries = df[cols_entries]
            #print(df_entries.head(15))#

            '''# check the values data
            #print(df.head(5))
            cols = df.columns.values
            n_cols = len(cols)
            print('nb of columns for histos : %d' % n_cols)
            cols_entries = cols[6::2]
            df_entries = df[cols_entries]
            #print(df_entries.head(15))#'''

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
                    totalDiff.append(DB.diffMAXKS(series0, series1, sum0, sum1)[0]) # 9000, 9000

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
                    totalDiff2.append(DB.diffMAXKS(series0, series_reference, sum0, nbBins_reference)[0]) # 9000, 9000

            print('ttl nb of couples 2 : %d' % nb2)
            
            # create the datas for the p-Value graph
            # by comparing the new curve with the others.
            # Get the new as reference (KS 3)
            nb3 = 0
            totalDiff3 = []
            for k in range(0,Nrows-0):
                nb3 += 1
                series0 = df_entries.iloc[k,:]
                sum0 = df_GetEntries[k]
                totalDiff3.append(DB.diffMAXKS(series0, s_new, sum0, Ntot_h_rel)[0])

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
            fig.savefig(folderKS + '/cumulative_curve_' + branches[i] + "_" + rel + '.png')
            fig.clf()
        
            # ================================ #
            # create the mean curve of entries #
            # ================================ #
            mean_df_entries = df_entries.mean()
            mean_sum = mean_df_entries.sum()
                
            diffMax1, posMax1 = DB.diffMAXKS(mean_df_entries, s_new, mean_sum, Ntot_h_rel)
            diffMax2, posMax2 = DB.diffMAXKS(series_reference, s_new, nbBins_reference, Ntot_h_rel)
            diffMax3, posMax3 = DB.diffMAXKS(s_new, s_KSref, Ntot_h_rel, Ntot_h_KSref)
            print("diffMax1 : %f - posMax1 : %f" % (diffMax1, posMax1))
            print("diffMax2 : %f - posMax2 : %f" % (diffMax2, posMax2))
            print("diffMax3 : %f - posMax3 : %f" % (diffMax3, posMax3))
            print('Ntot_h_rel : %d - Ntot_h_KSref : %d' % (Ntot_h_rel, Ntot_h_KSref))

            # diff max between new & old
            diffMax0, posMax0, sDKS = DB.diffMAXKS2(s_KSref, s_new, Ntot_h_KSref, Ntot_h_rel)
            print("diffMax0 : %f - posMax0 : %f" % (diffMax0, posMax0))
            wKS0.write('%s : %e\n' % (branches[i], diffMax0))
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
            KSDiffname1 = folder + '/KSDiffValues_1_' + branches[i] + '.txt' # csv imposed by pd.to_csv + "_" + rel 
            df.to_csv(KSDiffname1)
            plt_diff_KS1 = seriesTotalDiff1.plot.hist(bins=nbins, title='KS diff. 1')
            print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.min()))
            print('\ndiffMax0/sTD.max 1 : %f/%f' % (diffMax0, seriesTotalDiff1.values.max()))
            if (diffMax0 >= seriesTotalDiff1.values.max()):
                color1 = 'r'
                nb_red1 += 1
                x1 = seriesTotalDiff1.values.max()
            elif (diffMax0 <= seriesTotalDiff1.values.min()):
                color1 = 'g'
                nb_green1 += 1
                x1 = seriesTotalDiff1.values.min()
            else:
                color1 = 'g'
                nb_green1 += 1
                x1 = diffMax0
            print('x1 : %f' % x1)
            ymi, yMa = plt_diff_KS1.get_ylim()
            plt_diff_KS1.vlines(x1, ymi, 0.9*yMa, color=color1, linewidth=4)
            fig = plt_diff_KS1.get_figure()
            fig.savefig(folderKS + '/KS-ttlDiff_1_' + branches[i] + "_" + rel + '.png')
            fig.clf()
            count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            KSDiffHistoname1 = folder + '/KSDiffHistoValues_1_' + branches[i] + "_" + rel + '.txt'
            wKSDiff1 = open(KSDiffHistoname1, 'w')
            wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in count))
            wKSDiff1.write('\n')
            wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in division))
            wKSDiff1.write('\n')
            wKSDiff1.close()

            # Get the max of the integral
            I_max = DB.integralpValue(division, count, 0.)
            ##print('\nMax. integral 1 : %0.4e for nbins=%d' % (I_max, nbins))
            # print the min/max values of differences
            ##print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x1))
            pValue = DB.integralpValue(division, count, diffMax0)
            #print(division)
            #print(count)
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
            plt_diff_KS2 = seriesTotalDiff2.plot.hist(bins=nbins, title='KS diff. 2')
            print('\ndiffMin0/sTD.min 1 : %f/%f' % (diffMax0, seriesTotalDiff2.values.min()))
            print('\ndiffMax0/sTD.max 2 : %f/%f' % (diffMax0, seriesTotalDiff2.values.max()))
            if (diffMax0 >= seriesTotalDiff2.values.max()):
                color2 = 'r'
                nb_red2 += 1
                x2 = seriesTotalDiff2.values.max()
            elif (diffMax0 <= seriesTotalDiff2.values.min()):
                color2 = 'g'
                nb_green2 += 1
                x2 = seriesTotalDiff2.values.min()
            else:
                color2 = 'g'
                nb_green2 += 1
                x2 = diffMax0
            print('x2 : %f' % x2)
            ymi, yMa = plt_diff_KS2.get_ylim()
            plt_diff_KS2.vlines(x2, ymi, 0.9*yMa, color=color2, linewidth=4)
            fig = plt_diff_KS2.get_figure()
            fig.savefig(folderKS + '/KS-ttlDiff_2_' + branches[i] + "_" + rel + '.png')
            fig.clf()
            count, division = np.histogram(seriesTotalDiff2, bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            KSDiffHistoname2 = folder + '/KSDiffHistoValues_2_' + branches[i] + "_" + rel + '.txt'
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
            plt_diff_KS3 = seriesTotalDiff3.plot.hist(bins=nbins, title='KS diff. 3')
            print('\ndiffMin0/sTD.min 3 : %f/%f' % (diffMax0, seriesTotalDiff3.values.min()))
            print('diffMax0/sTD.max 3 : %f/%f' % (diffMax0, seriesTotalDiff3.values.max()))
            if (diffMax0 >= seriesTotalDiff3.values.max()):
                color3 = 'r'
                nb_red3 += 1
                x3 = seriesTotalDiff3.values.max()
            elif (diffMax0 <= seriesTotalDiff3.values.min()):
                color3 = 'g'
                nb_green3 += 1
                x3 = seriesTotalDiff3.values.min()
            else:
                color3 = 'g'
                nb_green3 += 1
                x3 = diffMax0
            print('x3 : %f' % x3)
            ymi, yMa = plt_diff_KS3.get_ylim()
            plt_diff_KS3.vlines(x3, ymi, 0.9*yMa, color=color3, linewidth=4)
            fig = plt_diff_KS3.get_figure()
            fig.savefig(folderKS + '/KS-ttlDiff_3_' + branches[i] + "_" + rel + '.png')
            fig.clf()
            count, division = np.histogram(seriesTotalDiff3, bins=nbins)
            div_min = np.amin(division)
            div_max = np.amax(division)
            KSDiffHistoname3 = folder + '/KSDiffHistoValues_3_' + branches[i] + "_" + rel + '.txt'
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

            wKSp.write('%s, %e, %e, %e\n' % (branches[i], pValue/I_max, pValue2/I_max2, pValue3/I_max3))

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
        else:
            print('%s KO' % branches[i])

    # print nb of red/green lines
    print('KS 1 : %d red - %d green for %s' % (nb_red1, nb_green1, rel))
    print('KS 2 : %d red - %d green for %s' % (nb_red2, nb_green2, rel))
    print('KS 3 : %d red - %d green for %s' % (nb_red3, nb_green3, rel))
    nb_red = nb_red1 + nb_red2 + nb_red3
    nb_green = nb_green1 + nb_green2 + nb_green3
    print('KS ttl : %d red - %d green for %s' % (nb_red, nb_green, rel))
    wKS_.write('KS 1 : %d red - %d green for %s\n' % (nb_red1, nb_green1, rel))
    wKS_.write('KS 2 : %d red - %d green for %s\n' % (nb_red2, nb_green2, rel))
    wKS_.write('KS 3 : %d red - %d green for %s\n' % (nb_red3, nb_green3, rel))
    wKS_.write('KS ttl : %d red - %d green for %s\n' % (nb_red, nb_green, rel))

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

print("Fin !\n")

