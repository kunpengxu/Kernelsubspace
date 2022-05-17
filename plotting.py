import seaborn as sns
from matplotlib import pyplot as plt
import generate_data_Multi_windows as gen
import numpy as np
from sklearn.cluster import SpectralClustering
import Build_selfmatrix as build
from scipy.stats import zscore
import tool as tool
import pandas as pd
from more_itertools import locate


def series_variation(dynamic=None, window_sizes=None, name=None, root=None):
    fig = plt.figure(figsize=(5.7, 5.7), dpi=150)
    sns.set_style('ticks')
    axe = fig.add_subplot(111)
    axe.plot(window_sizes, [dynamic[s] for s in window_sizes],
             'k-o', lw=.7, alpha=1, markersize=3)
    tresh = gen.entropy_cut_off(np.array([dynamic[s] for s in window_sizes]))

    axe.vlines(x=tresh, ymin=0, ymax=max(list(dynamic.values())),
               color='red')
    axe.xaxis.set_tick_params(direction='in', top=False, bottom=True,
                              length=8, color='k', labelsize=12, pad=2)
    axe.yaxis.set_tick_params(direction='in', top=False, bottom=True,
                              length=8, color='k', labelsize=12, pad=2)
    axe.minorticks_on()

    axe.set_ylabel('Evaluated score', weight='semibold', fontsize=23)

    axe.set_xlabel('Window size', weight='semibold', fontsize=23)

    plt.xticks(weight='semibold', rotation=45)
    plt.yticks(weight='semibold')
    med = np.max(list(dynamic.values())) / 2
    axe.text(tresh / 5, med, 'Sparse\n area', color='darkgreen', fontsize=28, weight='semibold', rotation=45)
    axe.text(tresh*1.05, med*1.7, 'Best window \nsize = ' + str(tresh), color='red', fontsize=25, weight='semibold')
    axe.text(tresh + tresh / 4, med, 'Dense\n area', color='darkred', fontsize=28, weight='semibold',
             rotation=45)

    plt.savefig(root + '/images/' + name + '_window_size.png', bbox_inches='tight')
    plt.show()

def ablation(series,win_size, lam, gam,root,name):
    print('processing compare representation model')
    #series = series.iloc[0:78, :]
    print('processing KBDR')
    [Z_KBDR, Z1, Z2, d] = build.find_KBDR(series, win_size, lam, gam)
    print('processing SSC')
    [Z_SSC] = tool.find_ssc(series)
    print('processing LRR')
    [Z_LRR, Z3] = tool.find_lrr(series)
    print('processing BDR')
    [Z_BDR] = tool.find_Block(series, 5, win_size,lam,gam)

    fig = plt.figure(figsize=(10, 6))

    ax5 = fig.add_subplot(221)
    ax5.set_title('SSC', fontsize=15)
    sns.heatmap((Z_SSC > 0).astype(np.int_), annot=False, xticklabels=False, yticklabels=False,
                square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
    ax6 = fig.add_subplot(222)
    ax6.set_title('LRR', fontsize=15)
    sns.heatmap((Z_LRR > 0).astype(np.int_), annot=False, xticklabels=False, yticklabels=False,
                square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
    ax7 = fig.add_subplot(223)
    ax7.set_title('BDR', fontsize=15)
    sns.heatmap((Z_BDR > 0).astype(np.int_), annot=False, xticklabels=False, yticklabels=False,
                square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
    ax8 = fig.add_subplot(224)
    ax8.set_title('Our model', fontsize=15)
    sns.heatmap((Z_KBDR > 0).astype(np.int_), annot=False, xticklabels=False, yticklabels=False,
                square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
    plt.savefig(root + '/images/' + name + '_Compare_with_SOAT.png', bbox_inches='tight')
    plt.show()

def Z_matrix(Z,root,name):
    print('plot first 2 windows')
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Z$_1$',fontsize=15)
    z0 = sns.heatmap(Z[0], annot=False,xticklabels=False, yticklabels=False,
                     square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})

    plt.xticks(rotation=0)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Z$_2$',fontsize=15)
    z1 = sns.heatmap(Z[1], annot=False,xticklabels=False, yticklabels=False,
                     square=True, cmap="YlGnBu_r", cbar=False, cbar_kws={"shrink": .39})
    plt.xticks(rotation=0)

    plt.xticks(rotation=0)
    plt.savefig(root + '/images/' + name + '_First_2_matrix.png', bbox_inches='tight')
    plt.show()

def Multi_scale(series, data, i, name, root):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 2))
    plt.suptitle('Synthetic')

    ax1.plot(series.apply(zscore, axis=0)[i], label='Original series: S', color='r')

    ax1.legend()
    ax2.plot(data[1][:, i], label='Scale - Level 0: S$^{(0)}$', color='r')
    ax2.legend()

    ax3.plot(data[0][:, i], label='Scale - Level 1: S$^{(1)}$', color='r')
    ax3.legend()


    fig.savefig(root + '/images/' + name + '_multi_scale', bbox_inches='tight', dpi=150)
    plt.show()

def regime(split_series_global,Z,root,name):
    print('plot regimes')
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    y_pred = SpectralClustering(n_clusters=5, affinity='precomputed').fit_predict(Z[2])
    cluster = np.unique(y_pred)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 2))
    locals()['index_pos_list%s' % 1] = list(locate(y_pred.tolist(), lambda a: a == cluster[0]))
    ax1.plot(pd.DataFrame(split_series_global[2])[locals()['index_pos_list%s' % 1]].mean(axis=1), color='black',
             label='Regime' + str(1))
    ax1.legend(loc='upper right')
    ymajorFormatter = FormatStrFormatter('%1.1f')
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    for i in range(1, len(cluster)):
        locals()['index_pos_list%s' % (i + 1)] = list(locate(y_pred.tolist(), lambda a: a == cluster[i]))
        locals()['ax%s' % (i + 1)].plot(
            pd.DataFrame(split_series_global[2])[locals()['index_pos_list%s' % (i + 1)]].mean(axis=1), color='black',
            label='Regime' + str(i + 1))
        locals()['ax%s' % (i + 1)].legend(loc='upper right')
        locals()['ax%s' % (i + 1)].yaxis.set_major_formatter(ymajorFormatter)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.suptitle('Synthetic')
    plt.savefig(root + '/images/' + name + '_regimes.png', bbox_inches='tight')
    plt.show()

def predict(a,b,root,name):
    plt.figure(figsize=(17, 3))
    plt.plot(a.iloc[:, 313], label='S' + str(313) + '_predicted', color='b')
    plt.plot(b.iloc[:, 313], label='S' + str(313), color='black', linestyle='-.')
    plt.savefig(root + '/images/' + name + '_predict.png', bbox_inches='tight')
    plt.legend()
    plt.show()