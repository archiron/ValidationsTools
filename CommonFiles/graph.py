#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

def createLossPictures(branch, history_da, nb_epochs, fileName):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.suptitle('Histo Plots for ' + branch)

    plt.subplot(1, 2, 1)
    plt.plot(list(range(nb_epochs))
    ,history_da['train_loss']
    ,label="train loss", color='red', linestyle = 'dotted')
    plt.plot(list(range(nb_epochs))
    ,history_da['test_loss']
    ,label="test loss", color='blue')
    plt.legend()
    plt.xlabel('nb epoch')

    plt.subplot(1, 2, 2)
    plt.plot(list(range(nb_epochs))
    ,history_da['train_loss']
    ,label="train loss", color='red', linestyle = 'dotted')
    plt.plot(list(range(nb_epochs))
    ,history_da['test_loss']
    ,label="test loss", color='blue')
    plt.legend()
    plt.yscale("log")
    plt.xlabel('nb epoch')

    plt.tight_layout()
    plt.savefig(fileName)
    return

def createPredictedPictures(branch, Ncols, new, y_pred_new, old, y_pred_old, new_loss, old_loss, fileName):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.suptitle(branch)

    plt.subplot(1, 2, 1)
    plt.plot(list(range(Ncols)) ,new ,label="CMSSW_12_1_0_pre5", color='red', marker='s', linestyle = 'dotted')
    pred_new = y_pred_new.numpy()
    plt.plot(list(range(Ncols)) ,pred_new[0] ,label="pred.", color='blue')
    plt.legend()
    plt.title("new : loss = {0:1.5e}".format(new_loss))

    plt.subplot(1, 2, 2)
    plt.plot(list(range(Ncols)) ,old ,label="CMSSW_12_1_0_pre4", color='red', marker='s', linestyle = 'dotted')
    pred_old = y_pred_old.numpy()
    plt.plot(list(range(Ncols)) ,pred_old[0] ,label="pred.", color='blue')
    plt.legend()
    plt.title("old : loss = {0:1.5e}".format(old_loss))

    plt.tight_layout()
    plt.savefig(fileName)
    #plt.show()
    return

def creatPredPictLinLog(branch, Ncols, new, y_pred_new, new_loss, rel, fileName):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.suptitle(branch)

    plt.subplot(1, 2, 1)
    y_new = new.numpy()
    #plt.plot(list(range(Ncols)), y_new[0], label=rel, color='red', marker='s', linestyle = 'dotted')
    plt.step(list(range(Ncols)), y_new[0], where='mid', label=rel, color='red', marker='s', linestyle = 'dotted')
    pred_new = y_pred_new.numpy()
    #plt.plot(list(range(Ncols)), pred_new[0], label="pred.", color='blue')
    plt.step(list(range(Ncols)), pred_new[0], where='mid', label="pred.", color='blue')
    plt.legend()
    plt.title("new : loss = {0:1.5e}".format(new_loss))

    plt.subplot(1, 2, 2)
    #plt.plot(list(range(Ncols)), y_new[0], label=rel, color='red', marker='s', linestyle = 'dotted')
    plt.step(list(range(Ncols)), y_new[0], where='mid', label=rel, color='red', marker='s', linestyle = 'dotted')
    pred_new = y_pred_new.numpy()
    #plt.plot(list(range(Ncols)), pred_new[0], label="pred.", color='blue')
    plt.step(list(range(Ncols)), pred_new[0], where='mid', label="pred.", color='blue')
    plt.legend()
    plt.title("new : loss = {0:1.5e}".format(new_loss))
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(fileName)
    return

def creatPredPictLin(branch, Ncols, new, y_pred_new, new_loss, rel, fileName):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.suptitle(branch)

    plt.subplot(1, 2, 1)
    y_new = new.numpy()
    #plt.plot(list(range(Ncols)), y_new[0], label=rel, color='red', marker='s', linestyle = 'dotted')
    plt.step(list(range(Ncols)), y_new[0], where='mid', label=rel, color='red', marker='s', linestyle = 'dotted')
    pred_new = y_pred_new.numpy()
    #plt.plot(list(range(Ncols)), pred_new[0], label="pred.", color='blue')
    plt.step(list(range(Ncols)), pred_new[0], where='mid', label="pred.", color='blue')
    plt.legend()
    plt.title("new : loss = {0:1.5e}".format(new_loss))

    plt.tight_layout()
    plt.savefig(fileName)
    return

def createCompPicture(branch, Ncols, new, ref, ref1, ref2, fileName):
    plt.clf()
    plt.step(list(range(Ncols)), new, where='mid', label=ref1, color='red', marker='*', linestyle = 'None')
    plt.step(list(range(Ncols)), ref, where='mid', label=ref2, color='blue')
    plt.legend()
    plt.title(branch)
    plt.savefig(fileName)
    return

def createMapPicture(X, Y, tab, Labels, fileName):
    plt.figure(figsize=(15, 15))
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, tab)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('releases')
    ax.set_ylabel('releases')
    ax.set_xticks(np.arange(len(Labels)))
    ax.set_xticklabels(Labels)
    ax.set_yticks(np.arange(len(Labels)))
    ax.set_yticklabels(Labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(fileName)
    fig.clf()
    return

def createCompLossesPicture(labels, val, fileName, title):
    x_pos = np.arange(len(labels))
    plt.clf()
    plt.figure(figsize=(10, 5))
    title = title.replace("_", "\_")
    plt.suptitle(title, x=0.35)

    plt.subplot(1, 2, 1)
    plt.plot(x_pos, val, color='blue', marker='*', linestyle = 'None')
    plt.ylabel('loss value')
    plt.xticks(x_pos, labels, rotation=45, ha="right", rotation_mode="anchor")

    plt.subplot(1, 2, 2)
    plt.plot(labels, val, color='blue', marker='*', linestyle = 'None')
    plt.xticks(x_pos, labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.yscale("log")
 
    plt.tight_layout()
    plt.savefig(fileName)
    return

def createCompLossesPicture2Axis(labs, val1, val2, fileName, title):
    x_pos = np.arange(len(labs))
    plt.clf()
    plt.figure(figsize=(10, 5))
    title = title.replace("_", "\_")

    fig, ax1 = plt.subplots()
    #ax1.set_xlabel('Releases') 
    #ax1.set_ylabel('loss value', color = 'red')
    ax1.set_title(title, x=0.50, y=1.05)
    plot_1 = ax1.plot(x_pos, val1, color='red', marker='*', linestyle = 'None', label='Loss value')
    ax1.tick_params(axis ='y', labelcolor = 'red') 

    plt.xticks(rotation=90)

    ax2 = ax1.twinx()
    #ax2.set_ylabel('KS Values', color = 'blue')
    locs = ax2.set_xticks(x_pos, labs)#, rotation=45, ha="right", rotation_mode="anchor")
    plot_2 = ax2.plot(x_pos, val2, color='blue', marker='+', linestyle = 'None', label='KS value')
    ax2.tick_params(axis ='y', labelcolor = 'blue') 

    lns = plot_1 + plot_2
    labels2 = [l.get_label() for l in lns]
    plt.legend(lns, labels2, loc=0)
    
    fig.tight_layout()
    plt.savefig(fileName)
    return

def createCompPValuesPicture(labels, val, fileName, title):
    x_pos = np.arange(len(labels))
    plt.clf()
    plt.figure(figsize=(10, 5))
    title = title.replace("_", "\_")
    plt.suptitle(title, x=0.35)
    #print(val)
    val1 = []
    val2 = []
    val3 = []
    N = len(val)
    #print('N={:d}'.format(N))
    for n in range(0,N-2,3):
        #print('{:d}/{:d}'.format(n,N-1))
        #print(val[n],val[n+1],val[n+2])
        val1.append(val[n])
        val2.append(val[n+1])
        val3.append(val[n+2])
    #print(val1)

    plt.subplot(1, 2, 1)
    plt.plot(x_pos, val1, color='blue', marker='*', linestyle = 'None', label='KS 1')
    plt.plot(x_pos, val2, color='green', marker='+', linestyle = 'None', label='KS 2')
    plt.plot(x_pos, val3, color='black', marker='o', linestyle = 'None', label='KS 3')
    plt.ylabel('max. diff.')
    plt.xticks(x_pos, labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(labels, val1, color='blue', marker='*', linestyle = 'None')
    plt.plot(x_pos, val2, color='green', marker='+', linestyle = 'None')
    plt.plot(x_pos, val3, color='black', marker='o', linestyle = 'None')
    plt.xticks(x_pos, labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.yscale("log")
 
    plt.tight_layout()
    plt.savefig(fileName)
    return

def createLatentPicture(labels,x,y, pictureName, title):
    # print only the latent positions for each release
    plt.clf()
    title = title.replace("_", "\_")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(x, y)
    ax1.set_xlabel('dim 1')
    ax1.set_xlabel('dim 2')
    ax1.set_title(title)
    
    for ind, text in enumerate(labels):
        ax1.annotate(text, (x[ind], y[ind]), xytext=(2,2), textcoords='offset points')
 
    fig.tight_layout()
    plt.savefig(pictureName)
    return

def createLatentPictureTrainTest(x_tr,y_tr,x_te,y_te, pictureName, title):
    # print the latent positions for each epoch
    plt.clf()
    title = title.replace("_", "\_")
    N = len(x_tr)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(x_tr, y_tr, color='red')
    ax1.scatter(x_te, y_te, color='blue', marker='+')
    ax1.set_xlabel('dim 1')
    ax1.set_xlabel('dim 2')
    ax1.set_title(title)
    ax1.annotate('0', (x_tr[0], y_tr[0]), xytext=(10,10), textcoords='offset points')
    ax1.annotate(N-1, (x_tr[N-1], y_tr[N-1]), xytext=(10,10), textcoords='offset points')
    plt.legend()
    fig.tight_layout()
    plt.savefig(pictureName)
    return

def createCompLatentPictureTrainTest(labels, x_tr,y_tr,x,y, pictureName, title):
    # print the latent positions for each epoch and the latent positions for each release
    plt.clf()
    title = title.replace("_", "\_")
    N = len(x_tr)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(x_tr, y_tr, color='red')
    ax1.scatter(x, y, color='blue')
    ax1.set_xlabel('dim 1')
    ax1.set_xlabel('dim 2')
    ax1.set_title(title)
    ax1.annotate('0', (x_tr[0], y_tr[0]), xytext=(10,10), textcoords='offset points')
    ax1.annotate(N-1, (x_tr[N-1], y_tr[N-1]), xytext=(10,10), textcoords='offset points')
    for ind, text in enumerate(labels):
        ax1.annotate(text, (x[ind], y[ind]), xytext=(10,10), textcoords='offset points')
    #plt.legend()
    fig.tight_layout()
    plt.savefig(pictureName)
    return

def createCompKSvsAEPicture(labels, val1, val2, fileName, title):
    x_pos = np.arange(len(labels))
    plt.clf()
    plt.figure(figsize=(10, 5))
    title = title.replace("_", "\_")
    plt.suptitle(title, x=0.35)

    plt.subplot(1, 2, 1)
    plt.plot(x_pos, val1, color='blue', marker='*', linestyle = 'None', label='KS values')
    plt.plot(x_pos, val2, color='green', marker='+', linestyle = 'None', label='AE values')
    plt.ylabel('max. diff.')
    plt.xticks(x_pos, labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(labels, val1, color='blue', marker='*', linestyle = 'None')
    plt.plot(x_pos, val2, color='green', marker='+', linestyle = 'None')
    plt.xticks(x_pos, labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.yscale("log")
 
    plt.tight_layout()
    plt.savefig(fileName)
    return

def createCompKSvsAEPicture2Axis(labels, val1, val2, fileName, title):
    x_pos = np.arange(len(labels))
    plt.clf()
    plt.figure(figsize=(10, 5))
    title = title.replace("_", "\_")
    #plt.suptitle(title, x=0.35)

    fig, ax1 = plt.subplots()
    #plt.subplot(1, 2, 1)
    ax1.set_title(title, x=0.50, y=1.05)
    plot_1 = ax1.plot(x_pos, val1, color='blue', marker='*', linestyle = 'None', label='KS values')
    ax1.tick_params(axis ='y', labelcolor = 'blue') 

    plt.xticks(rotation=90)
    
    ax2 = ax1.twinx()
    locs = ax2.set_xticks(x_pos, labels)#, rotation=45, ha="right", rotation_mode="anchor")
    plot_2 = ax2.plot(x_pos, val2, color='green', marker='+', linestyle = 'None', label='AE values')
    ax2.tick_params(axis ='y', labelcolor = 'green') 

    lns = plot_1 + plot_2
    labels2 = [l.get_label() for l in lns]
    plt.legend(lns, labels2, loc=0)

    plt.tight_layout()
    plt.savefig(fileName)
    return

