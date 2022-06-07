#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# zeeExtract_MP: a tool to generate Kolmogorov-Smirnov values/pictures
# for egamma validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys
import multiprocessing
import time

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv
from os import listdir
from os.path import isfile, join

argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from ROOT import *

ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

sys.path.append('../ChiLib_CMS_Validation')
import default as df
from graphicFunctions import getHisto
from default import *
from DecisionBox import DecisionBox
from sources import *

# these line for daltonians !
#seaborn.set_palette('colorblind')

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def changeColor(color):
    # 30:noir ; 31:rouge; 32:vert; 33:orange; 34:bleu; 35:violet; 36:turquoise; 37:blanc
    # other references at https://misc.flogisoft.com/bash/tip_colors_and_formatting
    if (color == 'black'):
        return '[30m'
    elif (color == 'red'):
        return '[31m'
    elif (color == 'green'):
        return '[32m'
    elif (color == 'orange'):
        return '[33m'
    elif (color == 'blue'):
        return '[34m'
    elif (color == ''):
        return '[35m'
    elif (color == 'purple'):
        return '[36m'
    elif (color == 'turquoise'):
        return '[37m'
    elif (color == 'lightyellow'):
        return '[93m'
    else:
        return '[30m'

def colorText(sometext, color):
    return '\033' + changeColor(color) + sometext + '\033[0m'

def changeDirectory(rootFile, path):
    """
    Change the current directory (ROOT.gDirectory) by the corresponding (rootFile,pathSplit)
    module from cmdLineUtils.py
    """
    rootFile.cd()
    theDir = ROOT.gDirectory.Get(path)
    if not theDir:
        print("Directory %s does not exist." % path)
    else:
        theDir.cd()
    return 0

def checkLevel(f_rel, f_out, path0, listkeys, nb, inPath):
    print('\npath : %s' % path0)
    if path0 != "":
        path0 += '/'
    for elem in listkeys:
        #print('%d == checkLevel : %s' % (nb, elem.GetTitle()))
        if (elem.GetClassName() == "TDirectoryFile"):
            path = path0 + elem.GetName()
            if (nb >= 3 and re.search(inPath, path)):
                print('\npath : %s' % path)
                f_out.mkdir(path)
            tmp = f_rel.Get(path).GetListOfKeys()
            checkLevel(path, tmp, nb+1)
        elif (elem.GetClassName() == "TTree"):
            #print('------ TTree')
            src = f_rel.Get(path0)
            cloned = src.CloneTree()
            #f_out.WriteTObject(cloned, elem.GetName())
            if (nb >= 3 and re.search(inPath, path0)):
                changeDirectory(f_out, path0[:-1])
                cloned.Write()
        elif (elem.GetClassName() != "TDirectory"):
            #print('copy %s object into %s path' % (elem.GetName(), path0[:-1]))
            #f_out.WriteTObject(elem.ReadObj(), elem.GetName())#:"DQMData/Run 1/EgammaV"
            if (nb >= 3 and re.search(inPath, path0)):
                changeDirectory(f_out, path0[:-1])
                elem.ReadObj().Write()

def getListFiles(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(".root")] # keep only root files
    return onlyfiles

def getBranches(t_p):
    b = []
    source = open("../ChiLib_CMS_Validation/HistosConfigFiles/ElectronMcSignalHistos.txt", "r")
    for ligne in source:
        if t_p in ligne:
            tmp = ligne.split(" ", 1)
            b.append(tmp[0].replace(t_p + "/", ""))
    source.close()
    return b

def diffR2(s0,s1):
    s0 = np.asarray(s0) # if not this, ind is returned as b_00x instead of int value
    s1 = np.asarray(s1)
    N = len(s0)
    R0 = 0.
    for i in range(0, N):
        t0 = s0[i]- s1[i]
        R0 += t0 * t0
    return R0/N

def getposColo(diffMax0, Diff_min, Diff_max):
    nb_red = 0
    nb_green = 0
    if (diffMax0 >= Diff_max):
        color = 'r'
        nb_red = 1
        x = Diff_max
    elif (diffMax0 <= Diff_min):
        color = 'g'
        nb_green = 1
        x = Diff_min
    else:
        color = 'g'
        nb_green = 1
        x = diffMax0
    return nb_red, nb_green, x, color

def cleanBranches(branches):
    #if (branches[i] == 'h_ele_seedMask_Tec'): # temp (pbm with nan)
    #if re.search('OfflineV', branches[i]): # temp (pbm with nbins=81 vs nbins=80)
    toBeRemoved = ['h_ele_seedMask_Tec'] # , 'h_ele_convRadius', 'h_ele_PoPtrue_golden_barrel', 'h_ele_PoPtrue_showering_barrel'
    for ele in toBeRemoved:
        if ele in branches:
            branches.remove(ele)

def func_Extract(br, nbFiles): # read files
    print("func_Extract")
    df.folderName = checkFolderName(df.folderName)

    branches = []
    wr = []
    histos = {}
        

    # get the branches for ElectronMcSignalHistos.txt
    #branches += ["h_recEleNum", "h_scl_ESFrac_endcaps", "h_scl_sigietaieta", "h_ele_PoPtrue_endcaps", "h_ele_PoPtrue", "h_scl_bcl_EtotoEtrue_endcaps", "h_scl_bcl_EtotoEtrue_barrel", "h_ele_Et"]
    #branches += ["h_recEleNum"]
    branches = br
    for leaf in branches:
        histos[leaf] = []
    
    fileList = getListFiles(folderName) # get the list of the root files in the folderName folder
    fileList.sort()
    print('there is %d files' % len(fileList))
    fileList = fileList[0:nbFiles]
    print('file list :')
    print(fileList)
    print('-- end --')

    for elem in fileList:
        input_file = folderName + str(elem.split()[0])
        name_1 = input_file.replace(folderName, '').replace('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_', '').replace('.root', '')
        print('\n %s - name_1 : %s' % (input_file, colorText(name_1, 'lightyellow')))
        
        f_root = ROOT.TFile(input_file) # 'DATA/' + 
        h1 = getHisto(f_root, tp_1)
        #h1.ls()
        for leaf in branches:
            print("== %s ==" % leaf)
            temp_leaf = []
            histo = h1.Get(leaf)

            temp_leaf.append(histo.GetMean()) # 0
            temp_leaf.append(histo.GetMeanError()) # 2
            temp_leaf.append(histo.GetStdDev()) # 6
            temp_leaf.append(histo.GetEntries()) # 6b

            temp_leaf.append(name_1) # 7
            #print('temp_leaf : %s' % temp_leaf)
            
            texttoWrite = ''
            i=0
            for entry in histo:
                #print(i,entry)
                texttoWrite += 'b_' + '{:03d}'.format(i) + ',c_' + '{:03d},'.format(i)
                temp_leaf.append(entry) # b_
                temp_leaf.append(histo.GetBinError(i)) # c_
                i+=1
            print('there is %d entries' % i)
            texttoWrite = texttoWrite[:-1] # remove last char
            temp_leaf.append(texttoWrite) # end
            histos[leaf].append(temp_leaf)

    #print histos into histo named files
    i_leaf = 0
    for leaf in branches:
        wr.append(open(folderName + 'histo_' + str(leaf) + '_' + '{:03d}'.format(nbFiles) + '_0_lite.txt', 'w'))
        nb_max = len(histos[leaf][0]) - 1
        print("== %s == nb_max : %d" % (leaf, nb_max))
        wr[i_leaf].write('evol,Mean,MeanError,StdDev,nbBins,name,')
        wr[i_leaf].write(str(histos[leaf][0][nb_max]))
        wr[i_leaf].write('\n')
        for i_file in range(0, len(fileList)):
            texttoWrite = str(i_file) + ','
            wr[i_leaf].write(texttoWrite) 
            for i in range(0, nb_max-1):
                #print('i : %d' % i)
                wr[i_leaf].write(str(histos[leaf][i_file][i]))
                wr[i_leaf].write(',')
            wr[i_leaf].write(str(histos[leaf][i_file][nb_max-1]))
            texttoWrite = '\n'
            wr[i_leaf].write(texttoWrite) 
        wr[i_leaf].close()
        i_leaf +=1
    return

def func_ReduceSize(nbFiles):
    print("func_ReduceSize")
    df.folderName = checkFolderName(df.folderName)

    fileList = getListFiles(folderName) # get the list of the root files in the folderName folder
    fileList.sort()
    print('there is %d files' % len(fileList))
    fileList = fileList[0:nbFiles]
    print('file list :')
    print(fileList)
    print('-- end --')

    for elem in fileList:
        input_file = folderName + str(elem.split()[0])
        print('\n %s' % input_file)

        paths = ['DQMData/Run 1/EgammaV', 'DQMData/Run 1/Info']

        f_rel = ROOT.TFile(input_file, "UPDATE")
        racine = input_file.split('.')
        f_out = TFile(racine[0] + 'b.' + racine[1], 'recreate')
        t2 = f_rel.GetListOfKeys()
        print(racine[0] + 'b.' + racine[1])
        for elem in paths:
            checkLevel(f_rel, f_out, "", t2, 0, elem)

        f_out.Close()
        f_rel.Close()

    return

def createKS_Curve(df, ttlD, yC1, yCC1, histo_name, diffMax0, nbins, str_nb):
    # Kolmogoroff-Smirnov curve
    DB = DecisionBox()
    df.folder = checkFolderName(df.folder)

    KSname1 = df.folder + "histo_" + histo_name + "_KScurve" + str_nb + ".txt"
    wKS1 = open(KSname1, 'w')      
    seriesTotalDiff = pd.DataFrame(ttlD, columns=['KSDiff'])
    KSDiffname1 = df.folder + '/KSDiffValues_' + str_nb + '_' + histo_name + '.txt'
    df.to_csv(KSDiffname1)
    plt_diff_KS1 = seriesTotalDiff.plot.hist(bins=nbins, title='KS diff.' + str_nb)
    print('\n%s :: diffMin0/sTD.min 1 : %f/%f' % (histo_name, diffMax0, seriesTotalDiff.values.min()))
    print('%s :: diffMax0/sTD.max 1 : %f/%f' % (histo_name, diffMax0, seriesTotalDiff.values.max()))
    aa = getposColo(diffMax0, seriesTotalDiff.values.min(), seriesTotalDiff.values.max())
    x1 = aa[2]
    color1 = aa[3]
    print('%s :: x : %f' % (histo_name, x1))
    ymi, yMa = plt_diff_KS1.get_ylim()
    plt_diff_KS1.vlines(x1, ymi, 0.9*yMa, color=color1, linewidth=4)
    fig = plt_diff_KS1.get_figure()
    fig.savefig(df.folder + '/KS-ttlDiff_' + str_nb + '_' + histo_name + '.png')
    fig.clf()
    count, division = np.histogram(seriesTotalDiff[~np.isnan(seriesTotalDiff)], bins=nbins)
    div_min = np.amin(division)
    div_max = np.amax(division)
    KSDiffHistoname1 = df.folder + '/KSDiffHistoValues_' + str_nb + '_' + histo_name + '.txt'
    wKSDiff1 = open(KSDiffHistoname1, 'w')
    wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in count))
    wKSDiff1.write('\n')
    wKSDiff1.write(' '.join("{:10.04e}".format(x) for x in division))
    wKSDiff1.write('\n')
    wKSDiff1.close()

    # Get the max of the integral
    I_max = DB.integralpValue(division, count, 0.)
    print('\n%s :: Max. integral : %0.4e for nbins=%d' % (histo_name, I_max, nbins))
    # print the min/max values of differences
    print('%s :: Kolmogoroff-Smirnov min value : %0.4e - max value : %0.4e | diff value : %e \n' % (histo_name, div_min, div_max, x1))
    #stop
    # print the p-Value
    pValue = DB.integralpValue(division, count, diffMax0)
    print('\n%s :: unormalized p_Value : %0.4e for nbins=%d' % (histo_name, pValue, nbins))
    print('%s :: normalized p_Value : %0.4e for nbins=%d' % (histo_name, pValue/I_max, nbins))

    # save the KS curves
    wKS1.write('%e, %d\n' % (I_max, nbins))
    wKS1.write('%e, %e\n' % (div_min, div_max))
    wKS1.write(' '.join("{:10.04e}".format(x) for x in count))
    wKS1.write('\n')
    wKS1.write(' '.join("{:10.04e}".format(x) for x in division))
    wKS1.write('\n')
    wKS1.write(' '.join("{:10.04e}".format(x) for x in yC1 )) # average (mean) curve
    wKS1.write('\n')
    wKS1.write(' '.join("{:10.04e}".format(x) for x in yCC1 ))
    wKS1.write('\n')
    wKS1.close()
    return aa, pValue/I_max

def createAll(args):
    df.folderName = checkFolderName(df.folderName)
    df.folder = checkFolderName(df.folder)
    DB = DecisionBox()
    #print(args)
    histo_name = args[0]
    histo_1 = args[1]
    histo_2 = args[2]
    nbins = args[3]
    ind_reference = args[4]
    name = df.folderName + "histo_" + histo_name + '_{:03d}'.format(nbFiles) + "_0_lite.txt"
    print('\n%s - %s' %(histo_name, name))
    df = pd.read_csv(name)
        
    print(histo_name) # print histo name
    #histo_1 = h1.Get(histo_name)
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
    #print(s_new)
    e_new = e_new[1:-1]
    Ntot_h1 = histo_1.GetEntries()

    #histo_2 = h2.Get(histo_name)
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
    Ntot_h2 = histo_2.GetEntries()
    print('%s :: Ntot_h1 : %d - Ntot_h2 : %d' % (histo_name, Ntot_h1, Ntot_h2))

    # print min/max for the new curve
    print('\n##########')
    print('%s :: min : %f' % (histo_name, s_new.min()))
    print('%s :: max : %f' % (histo_name, s_new.max()))
    print('###########\n')
    if (s_new.min() < 0.):
        print('%s :: pbm whith histo %s, min < 0' % (histo_name, histo_name))
        return
    if (np.floor(s_new.sum()) == 0.):
        print('%s :: pbm whith histo %s, sum = 0' % (histo_name, histo_name))
        return

    # check the values & errors data
    #print(df.head(5))
    cols = df.columns.values
    n_cols = len(cols)
    print('%s :: nb of columns for histos : %d' % (histo_name, n_cols))
    cols_entries = cols[6::2]
    df_entries = df[cols_entries]

    # nbBins (GetEntries())
    df_GetEntries = df['nbBins']

    # get nb of columns & rows for histos
    (Nrows, Ncols) = df_entries.shape
    print('%s :: [Nrows, Ncols] : [%d, %d]' % (histo_name, Nrows, Ncols))
    df_entries = df_entries.iloc[:, 1:Ncols-1]
    (Nrows, Ncols) = df_entries.shape
    print('%s :: [Nrows, Ncols] : [%d, %d]' % (histo_name, Nrows, Ncols))
    # we must verify if Nrows = nbFiles !

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

    print('%s :: ttl nb1 of couples 1 : %d' % (histo_name, nb1))

    # create the datas for the p-Value graph
    # by comparing 1 curve with the others.
        # Get a random histo as reference (KS 2)
        #ind_reference = 199 # np.random.randint(0, Nrows)
        #print('%s :: reference ind. : %d' % (histo_name, ind_reference))
    series_reference = df_entries.iloc[ind_reference,:]
    nbBins_reference = df_GetEntries[ind_reference]
    print('%s :: nb bins reference : %d' % (histo_name, nbBins_reference))
    #print(series_reference)
    nb2 = 0
    totalDiff2 = []
    for k in range(0,Nrows-0):
        if (k != ind_reference):
            nb2 += 1
            series0 = df_entries.iloc[k,:]
            sum0 = df_GetEntries[k]
            totalDiff2.append(DB.diffMAXKS(series0, series_reference, sum0, nbBins_reference)[0]) # 9000, 9000

    print('%s :: ttl nb of couples 2 : %d' % (histo_name, nb2))
    
    # create the datas for the p-Value graph
    # by comparing the new curve with the others.
    # Get the new as reference (KS 3)
    nb3 = 0
    totalDiff3 = []
    for k in range(0,Nrows-0):
        nb3 += 1
        series0 = df_entries.iloc[k,:]
        sum0 = df_GetEntries[k]
        totalDiff3.append(DB.diffMAXKS(series0, s_new, sum0, Ntot_h1)[0])

    print('%s :: ttl nb of couples 3 : %d' % (histo_name, nb3))
    
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
    fig.savefig(df.folder + '/cumulative_curve_' + histo_name + '.png')
    fig.clf()
    
    # ================================ #
    # create the mean curve of entries #
    # ================================ #
    mean_df_entries = df_entries.mean()
    mean_sum = mean_df_entries.sum()
    print('mean_sum : %d' % mean_sum)
    diffMax1, posMax1 = DB.diffMAXKS(mean_df_entries, s_new, mean_sum, Ntot_h1)
    diffMax2, posMax2 = DB.diffMAXKS(series_reference, s_new, nbBins_reference, Ntot_h1)
    diffMax3, posMax3 = DB.diffMAXKS(s_new, s_old, Ntot_h1, Ntot_h2)
    print("%s :: diffMax1 : %f - posMax1 : %f" % (histo_name, diffMax1, posMax1))
    print("%s :: diffMax2 : %f - posMax2 : %f" % (histo_name, diffMax2, posMax2))
    print("%s :: diffMax3 : %f - posMax3 : %f" % (histo_name, diffMax3, posMax3))
    print('%s :: Ntot_h1 : %d - Ntot_h2 : %d' % (histo_name, Ntot_h1, Ntot_h2))

    # diff max between new & old
    diffMax0, posMax0, sDKS = DB.diffMAXKS2(s_old, s_new, Ntot_h2, Ntot_h1)
    print("%s :: diffMax0 : %f - posMax0 : %f" % (histo_name, diffMax0, posMax0))
    #stop

    print("{}".format(histo_name)," ".join("{:10.04e}".format(x) for x in s_new[0:8]))
    print("{}".format(histo_name)," ".join("{:10.04e}".format(x) for x in s_old[0:8]))
    print("{}".format(histo_name)," ".join("{:10.04e}".format(x) for x in sDKS[0:8])) # diff

    yellowCurve1 = mean_df_entries
    yellowCurve2 = series_reference
    yellowCurve3 = s_new
    yellowCurveCum1 = DB.funcKS(mean_df_entries) #  cumulative yellow curve
    yellowCurveCum2 = DB.funcKS(series_reference)
    yellowCurveCum3 = DB.funcKS(s_new)

    # Kolmogoroff-Smirnov curve 1
    bb1, pV1 = createKS_Curve(df, totalDiff, yellowCurve1, yellowCurveCum1, histo_name, diffMax0, nbins, '1')
    nb_red1 = bb1[0]
    nb_green1 = bb1[1]
    x1 = bb1[2]
    print('%s :: x1 : %f - color : %s' % (histo_name, x1, bb1[3]))

    # Kolmogoroff-Smirnov curve 2
    bb2, pV2 = createKS_Curve(df, totalDiff2, yellowCurve2, yellowCurveCum2, histo_name, diffMax0, nbins, '2')
    nb_red2 = bb2[0]
    nb_green2 = bb2[1]
    x2 = bb2[2]
    print('%s :: x2 : %f - color : %s' % (histo_name, x2, bb2[3]))

    # Kolmogoroff-Smirnov curve 2
    bb3, pV3 = createKS_Curve(df, totalDiff3, yellowCurve3, yellowCurveCum3, histo_name, diffMax0, nbins, '3')
    nb_red3 = bb3[0]
    nb_green3 = bb3[1]
    x3 = bb3[2]
    print('%s :: x3 : %f - color : %s' % (histo_name, x3, bb3[3]))

    return histo_name, diffMax0, nb_red1, nb_green1, nb_red2, nb_green2, nb_red3, nb_green3, pV1, pV2, pV3

def func_CreateKS(br, nbFiles):
    print("func_Extract")
    df.folder = checkFolderName(df.folder)

    branches = br
    N_histos = len(branches)
    print('N_histos : %d' % N_histos)
    
    # nb of bins for sampling
    nbins = 100 
    
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

    # get the "new" root file datas
    f_rel = ROOT.TFile(input_rel_file)

    # get the "reference" root file datas
    f_ref = ROOT.TFile(input_ref_file)

    print('we use the %s file as reference' % input_ref_file)
    print('we use the %s file as new release' % input_rel_file)

    nb_red1 = 0
    nb_green1 = 0
    nb_red2 = 0
    nb_green2 = 0
    nb_red3 = 0
    nb_green3 = 0

    KS_diffName = df.folder + "histo_differences_KScurve.txt"
    print("KSname 1 : %s" % KS_diffName)
    wKS0 = open(KS_diffName, 'w')

    KS_resume = df.folder + "histo_resume.txt"
    print("KSname 0 : %s" % KS_resume)
    wKS_ = open(KS_resume, 'w')

    KS_pValues = df.folder + "histo_pValues.txt"
    print("KSname 2 : %s" % KS_pValues)
    wKSp = open(KS_pValues, 'w')

    h1 = getHisto(f_rel, tp_1)
    h2 = getHisto(f_ref, tp_1)

    # Get a random histo as reference (KS 2)
    ind_reference = 199 # np.random.randint(0, nbFiles)
    print('reference ind. : %d' % ind_reference)

    #######   MULTIPROCESSING   #######
    tic = time.time()
    print('===', h1)
    pool_obj = multiprocessing.Pool()
    args = ((elem, h1.Get(elem), h2.Get(elem), nbins, ind_reference) for elem in branches) 
    answer = pool_obj.map(createAll, args)
    print(answer, len(answer))
    # to remove None values in list
    answer2 = list(filter(None, answer))
    print(answer2, len(answer))
    for item in answer2:
        print(item[0], item[1], item[2], item[3])
        wKS0.write('%s : %e\n' % (item[0], item[1]))
        nb_red1 += item[2]
        nb_green1 += item[3]
        nb_red2 += item[4]
        nb_green2 += item[5]
        nb_red3 += item[6]
        nb_green3 += item[7]
        wKSp.write('%s, %e, %e, %e\n' % (item[0], item[8], item[9], item[10]))

    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))
    #stop

    # print nb of red/green lines
    print('')
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

    return

if __name__=="__main__":

    # get the branches for ElectronMcSignalHistos.txt
    branches = []
    branches = getBranches(tp_1)
    print(branches[0:9])
    #branches = branches[151:200]
    #branches = ['h_ele_PoPtrueVsEta_pfx', 'h_ele_PoPtrueVsPhi_pfx', 'h_scl_EoEtruePfVsEg_pfy', 'h_ele_EtaMnEtaTrueVsEta_pfx']
    cleanBranches(branches) # remove some histo wich have a pbm with KS.

    # nb of files to be used
    nbFiles = 750

    #func_ReduceSize(nbFiles)

    #func_Extract(branches[0:5], nbFiles) # create file with histo datas.
    func_Extract(branches, nbFiles) # create file with histo datas.

    #func_CreateKS(branches[230:], nbFiles) # create the KS files from histos datas for datasets
    #func_CreateKS(branches, nbFiles)  # create the KS files from histos datas

    print("Fin !")

