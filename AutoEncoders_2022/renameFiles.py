#!/usr/bin/env python
# coding: utf-8

import os
from re import I
listComp_av = []
listComp_ap = []

'''### FILES in COMP ... PNG
# Get the list of all files and directories
path = "/home/arnaud/cernbox/DEV_PYTHON/AutoEncoder/2022/RESULTS"
for root, dirs, files in os.walk(path, topdown=False):
    print(root)
    for name in files:
        fileName_av = os.path.join(root, name)
        #print(fileName_av)
        if 'comparison_' in fileName_av:
            listComp_av.append(fileName_av)
            print(fileName_av)
            fileName_ap = fileName_av.replace('_12_1_0_pre4', '')
            print(fileName_ap)
            listComp_ap.append(fileName_ap)
            os.rename(fileName_av, fileName_ap)
#   for name in dirs:
#      print(os.path.join(root, name))

print(listComp_av)
print(listComp_ap)'''

### FILES in html
path = "/home/arnaud/cernbox/DEV_PYTHON/AutoEncoder/2022/RESULTS"
for root, dirs, files in os.walk(path, topdown=False):
    print(root)
    for name in files:
        fileName_av = os.path.join(root, name)
        #print(fileName_av)
        if '.html' in fileName_av:
            Lones = []
            listComp_av.append(fileName_av)
            print(fileName_av)
            file1 = open(fileName_av, 'r')
            Lines = file1.readlines()
            file1.close()
            for item in Lines:
                if 'comparison_' in item:
                    if '_12_1_0_pre4' in item:
                        print(item)
                        item = item.replace('_12_1_0_pre4', '')
                        print(item)
                        Lones.append(item)
                    else:
                        Lones.append(item)
                else:
                    Lones.append(item)
            file1 = open(fileName_av, 'w')
            #file1.write(Lines)
            file1.writelines("%s\n" % i for i in Lones)
            file1.close()
#   for name in dirs:
#      print(os.path.join(root, name))

#print(listComp_av)
#print(listComp_ap)
