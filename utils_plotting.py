import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from utils import *
import configs

def recall_plots(
    cache_identification, narrow_recall, wide_recall,
    cache_states, recall_downsampling_idxs=None,
    fignum=4, save_name=None, ci_plot_width=1.75):
    
    
    cache_strs = [r'$\mathrm{C_1}$', r'$\mathrm{C_2}$', r'$\mathrm{C_3}$']
    cache_strs_colors = ['#95AAD2', '#155289', '#BAB4D8']
    
    num_states, N_bar = narrow_recall.shape
    threshold = 0.5
    if recall_downsampling_idxs is not None:
        cache_idxs_in_downsampling = recall_downsampling_idxs[1]
        recall_downsampling_idxs = recall_downsampling_idxs[0]
        narrow_recall = narrow_recall[:,recall_downsampling_idxs]
        wide_recall = wide_recall[:,recall_downsampling_idxs]
        N_bar = narrow_recall.shape[1]

    # Cache identification plot
    fig, ax = plt.subplots(figsize=(ci_plot_width,1.25))
    readout = cache_identification/cache_identification.max()
    ax.plot(readout, color='black')
    idxs = readout > threshold
    y = readout[idxs]
    y[y>0] = 1.05
    ax.scatter(np.arange(100)[idxs], y, s=1, color='red')
    xtick_loc = []; xtick_label = [];
    for i, c in enumerate(cache_states):
        xtick_loc.append(c)
        xtick_label.append((cache_strs[i]))
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=6)
    xticks = ax.get_xticklabels()
    for i in range(len(xticks)):
        xticks[i].set_color(cache_strs_colors[i])
    ax.set_yticks([0, 0.50, 1.0])
    ax.set_yticklabels([0, 0.50, 1.0], fontsize=6)
    ax.set_ylabel('Seed Output', fontsize=7)
    ax.set_xlabel('Current Place', fontsize=7)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'figures/fig{fignum}_seed_{save_name}.svg', dpi=300, transparent=True)
    plt.show()
    
    # Recall plots
    fig, ax = plt.subplots(figsize=(1.25,1.25))
    ax.imshow(narrow_recall.T, aspect=num_states/N_bar, cmap='binary',)
    ytick_loc = []; tick_label = [];
    for i, c in enumerate(cache_states):
        if recall_downsampling_idxs is not None:
            ytick_loc.append(cache_idxs_in_downsampling[i])
        else:
            ytick_loc.append((c/num_states)*N_bar)
        tick_label.append(cache_strs[i])
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(tick_label, fontsize=6)
    yticks = ax.get_yticklabels()
    for i in range(len(yticks)):
        yticks[i].set_color(cache_strs_colors[i])
    ax.set_xticks(cache_states)
    ax.set_xticklabels(tick_label, fontsize=6)
    xticks = ax.get_xticklabels()
    for i in range(len(xticks)):
        xticks[i].set_color(cache_strs_colors[i])
    ax.set_ylabel('Place Output', fontsize=7)
    ax.set_xlabel('Current Place', fontsize=7)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'figures/fig{fignum}_place1_{save_name}.svg', dpi=300, transparent=True)
    plt.show()
    
    
    # Wide recall plot
    fig, ax = plt.subplots(figsize=(1.25,1.25))
    ax.imshow(wide_recall.T, aspect=num_states/N_bar, cmap='binary',)
    ytick_loc = []; tick_label = [];
    for i, c in enumerate(cache_states):
        if recall_downsampling_idxs is not None:
            ytick_loc.append(cache_idxs_in_downsampling[i])
        else:
            ytick_loc.append((c/num_states)*N_bar)
        tick_label.append(cache_strs[i])
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(tick_label, fontsize=6)
    yticks = ax.get_yticklabels()
    for i in range(len(yticks)):
        yticks[i].set_color(cache_strs_colors[i])
    ax.set_xticks(cache_states)
    ax.set_xticklabels(tick_label, fontsize=6)
    xticks = ax.get_xticklabels()
    for i in range(len(xticks)):
        xticks[i].set_color(cache_strs_colors[i])
    ax.set_ylabel('Place Output', fontsize=7)
    ax.set_xlabel('Current Place', fontsize=7)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'figures/fig{fignum}_place2_{save_name}.svg', dpi=300, transparent=True)
    plt.show()