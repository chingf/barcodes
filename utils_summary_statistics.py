import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import os
from math import floor, ceil
import configs
from utils import *

def get_dist_from_attractor(cache_state_idxs):
    num_states = cache_state_idxs.size
    cache_states = np.argwhere(cache_state_idxs)
    dists = np.zeros(cache_state_idxs.size)
    counts = np.zeros(cache_state_idxs.size)
    states = []
    for i in range(dists.size):
        d = np.array([distance(i, c, num_states) for c in cache_states])
        d = d.flatten()
        dists[i] = d.min()
        counts[i] = np.sum(d==d.min())
        s = cache_states[d==d.min()].flatten()
        states.append(s)
    return dists, counts, states

def get_resolution_summary_statistics(
        readout, reconstruct, cache_states,
        activations, inputs,
        site_spacing, search_strength):

    num_states = readout.size
    N_inp = N_bar = reconstruct.shape[1]

    identification_1 = { # What threshold to binarize at?
        'threshold': [], 'site spacing': [], 'search strength': [],
        'accuracy': [], 'sensitivity': [], 'specificity': []
        }

    identification_2 = { # What are noncache values b/n cache 1 & 2?
        'site spacing': [], 'search strength': [],
        'noncache diff': [], 'noncache val': []
        }

    identification_3 = { # What are noncache vals away from cache locs?
        'site spacing': [], 'search strength': [],
        'dist from attractor': [], 'val': [],
        }
    
    identification_4 = { # What are vals at each cache??
        'site spacing': [], 'search strength': [],
        'cache': [], 'val': [],
        }
    
    identification_5 = { # What are vals at each cache??
        'site spacing': [], 'search strength': [],
        'n_caches_correct': [],
        }

    reconstruct_1 = { # Validity
        'p_valid': [], 'search strength': [],
        'site spacing': [], 'opt attractor dist': []
        }

    reconstruct_2 = { # Conditioned on validity, what is the norm error?
        'norm error': [], 'search strength': [],
        'opt attractor dist': [], 'site spacing': [],
        'chosen attractor dist': []
        }

    reconstruct_3 = { # Conditioned on validity, what is the norm error b/n cache 1 & 2?
        'norm error': [], 'search strength': [],
        'site spacing': [], 'peak error': [],
        }

    activations_1 = {
        'cache corr': [], 'distance': []
        }
    
    activations_2 = {
        'place-cache corr': []
        }
    
    summary_stats = {
        'identification_1': identification_1, 'identification_2': identification_2,
        'identification_3': identification_3, 'identification_4': identification_4,
        'identification_5': identification_5,
        'reconstruct_1': reconstruct_1, 'reconstruct_2': reconstruct_2,
        'reconstruct_3': reconstruct_3, 'activations_1': activations_1
        }
    
    cache_state_idxs = np.zeros(readout.size).astype(bool)
    for c in cache_states:
         cache_state_idxs[c] = True
    noncache_state_idxs = np.logical_not(cache_state_idxs)
    dist_from_attractor, dist_counts, dist_states = \
        get_dist_from_attractor(cache_state_idxs)

    # Identification 1
    for threshold in np.arange(0.0, 1.0, 0.1):
        _readout = np.digitize(readout, [threshold])
        true_pos = np.sum(_readout[cache_state_idxs])
        true_neg = np.sum(_readout[noncache_state_idxs]==0)
        acc = (true_pos + true_neg)/100
        identification_1['threshold'].append(threshold)
        identification_1['site spacing'].append(site_spacing)
        identification_1['search strength'].append(search_strength)
        identification_1['accuracy'].append(acc)
        identification_1['sensitivity'].append(true_pos/3)
        identification_1['specificity'].append(true_neg/97)

    # Identification 2
    c1 = cache_states[0]; c2 = cache_states[1];
    cache_val = min([readout[c1], readout[c2]])
    nc1 = floor((c1 + c2)/2); nc2 = ceil((c1 + c2)/2);
    noncache_val = (readout[nc1] + readout[nc2])/2
    identification_2['noncache val'].append(noncache_val)
    identification_2['noncache diff'].append(noncache_val-cache_val)
    identification_2['site spacing'].append(site_spacing)
    identification_2['search strength'].append(search_strength)

    # Identification 3
    vals = readout[noncache_state_idxs]
    identification_3['site spacing'].extend([site_spacing]*97)
    identification_3['search strength'].extend([search_strength]*97)
    identification_3['dist from attractor'].extend(
        list(dist_from_attractor[noncache_state_idxs]))
    identification_3['val'].extend(list(vals))
    
    # Identification 4
    vals = readout[cache_state_idxs]
    identification_4['site spacing'].extend([site_spacing]*3)
    identification_4['search strength'].extend([search_strength]*3)
    identification_4['cache'].extend([1,2,3])
    identification_4['val'].extend(list(vals))
    
    # Identification 4
    n_caches_correct = np.sum(readout[cache_state_idxs] > 0.5)
    identification_5['site spacing'].append(site_spacing)
    identification_5['search strength'].append(search_strength)
    identification_5['n_caches_correct'].append(n_caches_correct)

    # Reconstruction 1
    peak_locs = np.argmax(reconstruct, axis=1)
    peak_locs = (peak_locs/N_inp)*num_states
    valid = np.logical_and(readout>0.5, np.isin(peak_locs, cache_states))
    reconstruct_1['p_valid'].extend(list(valid))
    reconstruct_1['search strength'].extend([search_strength]*100)
    reconstruct_1['site spacing'].extend([site_spacing]*100)
    reconstruct_1['opt attractor dist'].extend(list(dist_from_attractor))

    # Reconstruction 2
    norm_errors = []
    peak_errors = []
    for idx in range(num_states):
        r = reconstruct[idx]/reconstruct[idx].max()
        norm_error = [np.linalg.norm(r - inputs[s]/inputs[s].max()) for s in dist_states[idx]]
        peak_error = [distance(peak_locs[idx], s, num_states) for s in dist_states[idx]]
        norm_error = min(norm_error); peak_error = min(peak_error)
        norm_errors.append(norm_error)
        peak_errors.append(peak_error)
        if not valid[idx]: continue
        reconstruct_2['norm error'].append(norm_error)
        reconstruct_2['search strength'].append(search_strength)
        reconstruct_2['opt attractor dist'].append(dist_from_attractor[idx])
        reconstruct_2['site spacing'].append(site_spacing)
        reconstruct_2['chosen attractor dist'].append(peak_error)

    # Reconstruction 3
    nc1 = floor((c1 + c2)/2); nc2 = ceil((c1 + c2)/2);
    norm_error = (norm_errors[nc1] + norm_errors[nc2])/2
    peak_error = (peak_errors[nc1] + peak_errors[nc2])/2
    reconstruct_3['norm error'].append(norm_error)
    reconstruct_3['peak error'].append(peak_error)
    reconstruct_3['search strength'].append(search_strength)
    reconstruct_3['site spacing'].append(site_spacing)
    
    # Activations 1 and 2
    if search_strength == 0:
        c = pairwise_correlations_centered(activations)
        distances = []; cache_corrs = []
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                distances.append(distance(i, j, num_states))
                cache_corrs.append(c[i,j])
        distances = np.array(distances); cache_corrs = np.array(cache_corrs)
        for _distance in np.unique(distances):
            idxs = distances==_distance
            _cache_corr = np.mean(cache_corrs[idxs])
            activations_1['cache corr'].append(_cache_corr)
            activations_1['distance'].append(_distance)
            
        acts_normalized = normalize(activations)
        inputs_normalized = normalize(inputs)
        p_c_corrs = [np.corrcoef(
            acts_normalized[i], inputs_normalized[i])[0, 1] for i in range(num_states)]
        activations_2['place-cache corr'].append(np.mean(p_c_corrs))
    
    return summary_stats
