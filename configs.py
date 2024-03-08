from dataclasses import dataclass
import numpy as np

fig_width = 6.4
fig_height = 4.8

n_states_per_site = 10

import seaborn as sns
sns.set(
        font_scale=14/12., #default size is 12pt, scale to 16pt
        palette='colorblind', #'Set1',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'text.color': 'dimgrey', #e.g. legend

            'lines.solid_capstyle': 'round',
            'lines.linewidth': 1,
            'legend.facecolor': 'white',
            'legend.framealpha':0.8,
            'legend.fontsize':8,
            'legend.title_fontsize':8,

            'xtick.bottom': True,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',

            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': True,

             'xtick.major.size': 2,
             'xtick.major.width': .5,
             'xtick.minor.size': 1,
             'xtick.minor.width': .5,

             'ytick.major.size': 2,
             'ytick.major.width': .5,
             'ytick.minor.size': 1,
             'ytick.minor.width': .5})
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']

import os
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/rl/'
else:
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/'
