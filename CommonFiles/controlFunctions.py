#!/usr/bin/env python
# coding: utf-8

################################################################################
# controlFunctions : a tool with all functions used in the KSTools environment
# for egamma AutoEncoder/KS validation comparison                              
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import os,sys

from os import listdir
from os.path import isfile, join

def checkFolderName(folderName):
    if folderName[-1] != '/':
        folderName += '/'
    return folderName

def checkFolder(folder):
    if not os.path.exists(folder): # create folder
        os.makedirs(folder) # create reference folder
    else: # folder already created
        print('%s already created.' % folder)
    return

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

def getListFiles(path, ext='root'):
    # use getListFiles(str path_where_the_files_are, str 'ext')
    # ext can be root, txt, png, ...
    # default is root
    ext = '.' + ext
    #print('path : %s' % path)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith(ext)] # keep only root files
    #print(onlyfiles)
    return onlyfiles

def getBranches(t_p, branchPath):
    b = []
    source = open(branchPath, "r")
    for ligne in source:
        if t_p in ligne:
            #print(ligne)
            tmp = ligne.split(" ", 1)
            #print(tmp[0].replace(t_p + "/", ""))
            b.append(tmp[0].replace(t_p + "/", ""))
    source.close()
    return b

def cleanBranches(branches):
    #if (branches[i] == 'h_ele_seedMask_Tec'): # temp (pbm with nan)
    #if re.search('OfflineV', branches[i]): # temp (pbm with nbins=81 vs nbins=80)
    toBeRemoved = ['h_ele_seedMask_Tec'] # , 'h_ele_convRadius', 'h_ele_PoPtrue_golden_barrel', 'h_ele_PoPtrue_showering_barrel'
    for ele in toBeRemoved:
        if ele in branches:
            branches.remove(ele)

def reduceBranch(branch):
    shn = branch.replace("h_", "").replace("ele_", "").replace("scl_", "").replace("bcl_", "")
    return shn

