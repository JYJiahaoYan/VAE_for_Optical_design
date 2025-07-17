import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colour
import torch

from sklearn.metrics import r2_score

def R2_plot_forward_color(color_raw, color_pred):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    titles = ['R', 'G', 'B']
    xlim = [[0.1,0.8],[0.0, 0.8],[0.0,0.7]]
    for i in range(3):
        raw_pred = np.array(sorted(zip(color_raw[:, i], color_pred[:, i])))
        print(raw_pred.shape)
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1], s =3 )
        ax[i].plot(xlim[i],xlim[i], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        ax[i].set_xlim(xlim[i])
        ax[i].set_ylim(xlim[i])
    plt.show()

def abs_err_plt_forward_color(color_raw, color_pred):
    abs_err = abs(color_raw - color_pred)
    abs_mean = sum(abs_err) / len(abs_err)
    print(abs_mean)
    plt.figure(figsize=[8, 14], dpi=150)
    plt.subplot(3, 1, 1)
    plt.scatter(color_raw[:, 0], abs_err[:, 0], color='r', label='R')
    plt.axhline(y=abs_mean[0], color='r', linestyle='-')
    plt.text(0.5, abs_mean[0], str(round(abs_mean[0], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.scatter(color_raw[:, 1], abs_err[:, 1], color='g', label='G')
    plt.axhline(y=abs_mean[1], color='g', linestyle='-')
    plt.text(0.6, abs_mean[1], str(round(abs_mean[1], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.scatter(color_raw[:, 2], abs_err[:, 2], color='b', label='B')
    plt.axhline(y=abs_mean[2], color='b', linestyle='-')
    plt.text(0.4, abs_mean[2], str(round(abs_mean[2], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

def R2_plot_forward_BIC(BIC_raw, BIC_pred):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
    titles = ['BIC']
    xlim = [[780,1200]]

    raw_pred = np.array(sorted(zip(BIC_raw[:, 0], BIC_pred[:, 0])))
    # print(raw_pred.shape)
    ax.scatter(raw_pred[:, 0], raw_pred[:, 1], s =3 )
    ax.plot(xlim[0],xlim[0], c='k')
    ax.set_title(titles[0] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
    ax.set_xlabel('ground truth')
    ax.set_ylabel('predicted')
    ax.set_xlim(xlim[0])
    ax.set_ylim(xlim[0])
    plt.show()

def abs_err_plt_forward_BIC(BIC_raw, BIC_pred):
    abs_err = abs(BIC_raw - BIC_pred)
    abs_mean = sum(abs_err) / len(abs_err)
    print(abs_mean)
    # plt.subplot(1, 1, 4)
    plt.scatter(BIC_raw[:, 0], abs_err[:, 0], color='yellow', label='BIC')
    plt.axhline(y=abs_mean[0], color='yellow', linestyle='-')
    plt.text(1050, abs_mean[0], str(round(abs_mean[0], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

def R2_plot(color_BIC_raw, color_BIC_pred):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), dpi=150)
    titles = ['R', 'G', 'B','BIC']
    xlim = [[0.1,0.8],[0.0, 0.8],[0.0,0.7],[780,1200]]
    for i in range(4):
        raw_pred = np.array(sorted(zip(color_BIC_raw [:, i], color_BIC_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1], s =3 )
        ax[i].plot(xlim[i],xlim[i], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        ax[i].set_xlim(xlim[i])
        ax[i].set_ylim(xlim[i])
    plt.show()

def abs_err_plt(color_BIC_raw, color_BIC_pred):
    abs_err = abs(color_BIC_raw - color_BIC_pred)
    abs_mean = sum(abs_err) / len(abs_err)
    print(abs_mean)
    plt.figure(figsize=[8, 14], dpi=150)
    plt.subplot(4, 1, 1)
    plt.scatter(color_BIC_raw[:, 0], abs_err[:, 0], color='r', label='R')
    plt.axhline(y=abs_mean[0], color='r', linestyle='-')
    plt.text(0.5, abs_mean[0], str(round(abs_mean[0], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.scatter(color_BIC_raw[:, 1], abs_err[:, 1], color='g', label='G')
    plt.axhline(y=abs_mean[1], color='g', linestyle='-')
    plt.text(0.6, abs_mean[1], str(round(abs_mean[1], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.scatter(color_BIC_raw[:, 2], abs_err[:, 2], color='b', label='B')
    plt.axhline(y=abs_mean[2], color='b', linestyle='-')
    plt.text(0.4, abs_mean[2], str(round(abs_mean[2], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.scatter(color_BIC_raw[:, 3], abs_err[:, 3], color='yellow', label='BIC')
    plt.axhline(y=abs_mean[3], color='yellow', linestyle='-')
    plt.text(1050, abs_mean[3], str(round(abs_mean[3], 4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()