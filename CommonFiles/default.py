#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# default.py : folders values to be used with zeeExtract tools
# for egamma validation comparison                              
# 
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

#############################################################################
# fixed data for zeeExtract for KS
tp_1 = 'ElectronMcSignalValidator'
nbFiles = 200
# be careful to not forget the '/' at the end of the path
#folder = 'Extraction_18/' # 200

#nbFiles = 950
#folder = 'Extraction_20h/' # 950

folder = 'TMP_Extraction_25/' # 200

# nb of bins for sampling
nbins = 100 

# reference index : -1=random integer (0<=index<=nbFiles)
ind_reference = 199

#############################################################################
