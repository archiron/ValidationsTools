#! /usr/bin/env python
#-*-coding: utf-8 -*-

################################################################################
# sources.py : list of ROOT files to be used with zeeExtract tools
# for egamma validation comparison                              
# 
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

# get the "new" root file datas
#input_rel_file = 'DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_new.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_11_3_0-113X_mcRun3_2021_realistic_v10-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre1-113X_mcRun3_2021_realistic_v10-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre2-113X_mcRun3_2021_realistic_v10-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre4-120X_mcRun3_2021_realistic_v2-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre3-120X_mcRun3_2021_realistic_v1-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre6-120X_mcRun3_2021_realistic_v4-v1__DQMIO.root'
#input_rel_file = 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_1_0_pre5-121X_mcRun3_2021_realistic_v15-v1__DQMIO.root'
input_rel_file = 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_5_0_pre5-125X_mcRun3_2022_realistic_v3-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_2_0_pre2-122X_mcRun3_2021_realistic_v1-v2__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_2_0_pre3-122X_mcRun3_2021_realistic_v5-v1__DQMIO.root'
#input_rel_file = 'DATA/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_056.root'

# get the "reference" root file datas
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_11_3_0_pre5-113X_mcRun3_2021_realistic_v7-v1__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_11_3_0_pre6-113X_mcRun3_2021_realistic_v9-v1__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre1-113X_mcRun3_2021_realistic_v10-v1__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre2-113X_mcRun3_2021_realistic_v10-v1__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre4-120X_mcRun3_2021_realistic_v2-v1__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_0_0_pre3-120X_mcRun3_2021_realistic_v1-v1__DQMIO.root'
#input_ref_file = 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_1_0_pre4-121X_mcRun3_2021_realistic_v10-v1__DQMIO.root'
#input_ref_file = 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_1_0_pre5-121X_mcRun3_2021_realistic_v15-v1__DQMIO.root'
input_ref_file = 'DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_5_0_pre4-124X_mcRun3_2022_realistic_v10-v1__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_1_0_pre5-121X_mcRun3_2021_realistic_v9000-v214__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__RelValZEE_14__CMSSW_12_2_0_pre2-122X_mcRun3_2021_realistic_v1-v2__DQMIO.root'
#input_ref_file = 'DATA/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_017.root'
#ind_ref_file = 214 # np.random.randint(0, nbFiles)
#input_ref_file = folderName + '/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_9000_' + '{:03d}'.format(ind_ref_file) + '.root'
#print('we use the %d file as reference' % ind_ref_file)
#print('we use : %s file as reference' % input_ref_file)
