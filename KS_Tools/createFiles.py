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
import imp
import time

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
from tracemalloc import stop

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 2 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 4 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
import matplotlib

# import matplotlib.dates as md
matplotlib.use('agg')
from matplotlib import pyplot as plt

blu = imp.load_source(filePaths, commonPath+filePaths)
print('DATA_SOURCE : %s' % blu.DATA_SOURCE)
resultPath = blu.RESULTFOLDER # checkFolderName(blu.RESULTFOLDER)
print('result path : {:s}'.format(resultPath))

Chilib_path = blu.LIB_SOURCE # checkFolderName(blu.LIB_SOURCE) # sys.argv[1]
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from controlFunctions import *
from graphicFunctions import getHisto
from DecisionBox import DecisionBox
from default import *
from sources import *

folder = checkFolderName(dfo.folder)
resultPath = checkFolderName(resultPath)

# get the branches for ElectronMcSignalHistos.txt
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = []
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

DB = DecisionBox()
print("\nextractFiles")

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

##### TEMP #####
LOG_SOURCE_WORK=blu.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/'
# get the "new" root file datas
f_rel = ROOT.TFile(LOG_SOURCE_WORK + input_rel_file)

# get the "reference" root file datas
f_ref = ROOT.TFile(LOG_SOURCE_WORK + input_ref_file)

print('we use the %s file as reference' % input_ref_file)
print('we use the %s file as new release' % input_rel_file)

nb_red1 = 0
nb_green1 = 0
nb_red2 = 0
nb_green2 = 0
nb_red3 = 0
nb_green3 = 0

KS_diffName = dfo.folder + "histo_differences_KScurve.txt"
print("KSname 1 : %s" % KS_diffName)
wKS0 = open(KS_diffName, 'w')

KS_resume = dfo.folder + "histo_resume.txt"
print("KSname 0 : %s" % KS_resume)
wKS_ = open(KS_resume, 'w')

KS_pValues = dfo.folder + "histo_pValues.txt" 
print("KSname 2 : %s" % KS_pValues)
wKSp = open(KS_pValues, 'w')

if (ind_reference == -1):
    ind_reference = np.random.randint(0, nbFiles)
print('reference ind. : %d' % ind_reference)

tic = time.time()
for i in range(0, N_histos): # 1 histo for debug
    name = resultPath + "histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
    print('\n%d - %s' %(i, name))

    df = pd.read_csv(name)
        
    h1 = getHisto(f_rel, tp_1)
    #print("h1")
    #print(h1)
    print(branches[i]) # print histo name
    histo_1 = h1.Get(branches[i])
    ii=0
    s_new = []
    e_new = []
    for entry in histo_1:
        #print("%d/%d : %s - %s") % (ii, histo_1.GetXaxis().GetNbins(), entry, histo_1.GetBinError(i))
        s_new.append(entry)
        e_new.append(histo_1.GetBinError(ii))
        ii += 1
    s_new = np.asarray(s_new)
    s_new = s_new[1:-1]
    e_new = e_new[1:-1]
    #print(s_new)
    #print(e_new)
    Ntot_h1 = histo_1.GetEntries()

    h2 = getHisto(f_ref, tp_1)
    #print("h2")
    #print(h2)
    print(branches[i]) # print histo name
    histo_2 = h2.Get(branches[i])
    ii=0
    s_old = []
    e_old = []
    for entry in histo_2:
        #print("%d/%d : %s - %s") % (ii, histo_2.GetXaxis().GetNbins(), entry, histo_2.GetBinError(i))
        s_old.append(entry)
        e_old.append(histo_2.GetBinError(ii))
        ii += 1
    s_old = np.asarray(s_old)
    s_old = s_old[1:-1]
    e_old = e_old[1:-1]
    #print(s_old)
    #print(e_old)
    Ntot_h2 = histo_2.GetEntries()
    print('Ntot_h1 : %d - Ntot_h2 : %d' % (Ntot_h1, Ntot_h2))

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
    KSname1 = dfo.folder + "histo_" + branches[i] + "_KScurve1.txt"
    KSname2 = dfo.folder + "histo_" + branches[i] + "_KScurve2.txt"
    KSname3 = dfo.folder + "histo_" + branches[i] + "_KScurve3.txt"
    print("KSname 1 : %s" % KSname1)
    print("KSname 2 : %s" % KSname2)
    print("KSname 3 : %s" % KSname3)
    wKS1 = open(KSname1, 'w')
    wKS2 = open(KSname2, 'w')
    wKS3 = open(KSname3, 'w')

    # check the values & errors data
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
            totalDiff.append(DB.diffMAXKS(series0, series1, sum0, sum1)[0]) # 9000, 9000

    print('ttl nb1 of couples 1 : %d' % nb1)

    # create the datas for the p-Value graph
    # by comparing 1 curve with the others.
    # Get a random histo as reference (KS 2)
        #ind_reference = np.random.randint(0, Nrows)
        #print('reference ind. : %d' % ind_reference)
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
    #print("s_new : ")
    #print(s_new)
    nb3 = 0
    totalDiff3 = []
    for k in range(0,Nrows-0):
        nb3 += 1
        series0 = df_entries.iloc[k,:]
        sum0 = df_GetEntries[k]
        totalDiff3.append(DB.diffMAXKS(series0, s_new, sum0, Ntot_h1)[0])

    print('ttl nb of couples 3 : %d' % nb3)
    
    # plot some datas (in fact doing nothing but creating fig)
    plt_entries = df_entries.plot(kind='line')
    fig = plt_entries.get_figure()
    fig.clf()
    # create the integrated curve
    curves = []
    for k in range(0,Nrows):
        series0 = df_entries.iloc[k,:]
        curves = DB.funcKS(series0)
        plt.plot(curves)
    fig.savefig(dfo.folder + '/cumulative_curve_' + branches[i] + '.png')
    fig.clf()
    
    # ================================ #
    # create the mean curve of entries #
    # ================================ #
    mean_df_entries = df_entries.mean()
    mean_sum = mean_df_entries.sum()
        
    diffMax1, posMax1 = DB.diffMAXKS(mean_df_entries, s_new, mean_sum, Ntot_h1)
    diffMax2, posMax2 = DB.diffMAXKS(series_reference, s_new, nbBins_reference, Ntot_h1)
    diffMax3, posMax3 = DB.diffMAXKS(s_new, s_old, Ntot_h1, Ntot_h2)
    print("diffMax1 : %f - posMax1 : %f" % (diffMax1, posMax1))
    print("diffMax2 : %f - posMax2 : %f" % (diffMax2, posMax2))
    print("diffMax3 : %f - posMax3 : %f" % (diffMax3, posMax3))
    print('Ntot_h1 : %d - Ntot_h2 : %d' % (Ntot_h1, Ntot_h2))

    # diff max between new & old
    diffMax0, posMax0, sDKS = DB.diffMAXKS2(s_old, s_new, Ntot_h2, Ntot_h1)
    print("diffMax0 : %f - posMax0 : %f" % (diffMax0, posMax0))
    wKS0.write('%s : %e\n' % (branches[i], diffMax0))
    print(s_new[0:8])
    print(s_old[0:8])
    print(sDKS[0:8]) # diff

    yellowCurve1 = mean_df_entries
    yellowCurve2 = series_reference
    yellowCurve3 = s_new
    yellowCurveCum1 = DB.funcKS(mean_df_entries) #  cumulative yellow curve
    yellowCurveCum2 = DB.funcKS(series_reference)
    yellowCurveCum3 = DB.funcKS(s_new)

    # Kolmogoroff-Smirnov curve
    seriesTotalDiff1 = pd.DataFrame(totalDiff, columns=['KSDiff'])
    KSDiffname1 = dfo.folder + '/KSDiffValues_1_' + branches[i] + '.txt' # csv imposed by pd.to_csv
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
    fig.savefig(dfo.folder + '/KS-ttlDiff_1_' + branches[i] + '.png')
    fig.clf()
    count, division = np.histogram(seriesTotalDiff1[~np.isnan(seriesTotalDiff1)], bins=nbins)
    div_min = np.amin(division)
    div_max = np.amax(division)
    KSDiffHistoname1 = dfo.folder + '/KSDiffHistoValues_1_' + branches[i] + '.txt'
    wKSDiff1 = open(KSDiffHistoname1, 'w')
    wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in count))
    wKSDiff1.write('\n')
    wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in division))
    wKSDiff1.write('\n')
    wKSDiff1.close()

    # Get the max of the integral
    I_max = DB.integralpValue(division, count, 0.)
    print('\nMax. integral 1 : %0.4e for nbins=%d' % (I_max, nbins))
    # print the min/max values of differences
    print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x1))
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
    fig.savefig(dfo.folder + '/KS-ttlDiff_2_' + branches[i] + '.png')
    fig.clf()
    count, division = np.histogram(seriesTotalDiff2, bins=nbins)
    div_min = np.amin(division)
    div_max = np.amax(division)
    KSDiffHistoname2 = dfo.folder + '/KSDiffHistoValues_2_' + branches[i] + '.txt'
    wKSDiff2 = open(KSDiffHistoname2, 'w')
    wKSDiff2.write(' '.join("{:10.04e}".format(x) for x in count))
    wKSDiff2.write('\n')
    wKSDiff2.write(' '.join("{:10.04e}".format(x) for x in division))
    wKSDiff2.write('\n')
    wKSDiff2.close()
    
    # Get the max of the integral
    I_max2 = DB.integralpValue(division, count, 0.)
    print('\nMax. integral 2 : %0.4e for nbins=%d' % (I_max2, nbins))
    # print the min/max values of differences
    print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x2))
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
    fig.savefig(dfo.folder + '/KS-ttlDiff_3_' + branches[i] + '.png')
    fig.clf()
    count, division = np.histogram(seriesTotalDiff3, bins=nbins)
    div_min = np.amin(division)
    div_max = np.amax(division)
    KSDiffHistoname3 = dfo.folder + '/KSDiffHistoValues_3_' + branches[i] + '.txt'
    wKSDiff3 = open(KSDiffHistoname3, 'w')
    wKSDiff3.write(' '.join("{:10.04e}".format(x) for x in count))
    wKSDiff3.write('\n')
    wKSDiff3.write(' '.join("{:10.04e}".format(x) for x in division))
    wKSDiff3.write('\n')
    wKSDiff3.close()
    
    # Get the max of the integral
    I_max3 = DB.integralpValue(division, count, 0.)
    print('\nMax. integral 3 : %0.4e for nbins=%d' % (I_max3, nbins))
    # print the min/max values of differences
    print('Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (div_min, div_max, x3))
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

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

# print nb of red/green lines
print('KS 1 : %d red - %d green' % (nb_red1, nb_green1))
print('KS 2 : %d red - %d green' % (nb_red2, nb_green2))
print('KS 3 : %d red - %d green' % (nb_red3, nb_green3))
nb_red = nb_red1 + nb_red2 + nb_red3
nb_green = nb_green1 + nb_green2 + nb_green3
print('KS ttl : %d red - %d green' % (nb_red, nb_green))
wKS_.write('KS 1 : %d red - %d green\n' % (nb_red1, nb_green1))
wKS_.write('KS 2 : %d red - %d green\n' % (nb_red2, nb_green2))
wKS_.write('KS 3 : %d red - %d green\n' % (nb_red3, nb_green3))
wKS_.write('KS ttl : %d red - %d green\n' % (nb_red, nb_green))

print("Fin !\n")

