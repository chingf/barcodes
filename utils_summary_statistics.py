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
    n_caches = len(cache_states)
    n_noncaches = num_states - n_caches

    identification_1 = { # What threshold to binarize at?
        'threshold': [], 'site spacing': [], 'search strength': [],
        'accuracy': [], 'sensitivity': [], 'specificity': [], 'n_caches': []
        }

    identification_2 = { # What are noncache values b/n cache 1 & 2?
        'site spacing': [], 'search strength': [],
        'noncache diff': [], 'noncache val': [], 'n_caches': []
        }

    identification_3 = { # What are vals at each cache??
        'site spacing': [], 'search strength': [],
        'cache': [], 'val': [], 'n_caches': []
        }   
    
    reconstruct_1 = { # Did you get the closest peak?
        'is_cache': [], 'high_readout': [], 'is_closest': [],
        'site spacing': [], 'distance from closest cache': [], 'n_caches': [],
        'search strength': []
        }

    reconstruct_2 = { # Conditioned on getting the closest peak, what is the norm error?
        'norm error': [], 'search strength': [],
        'distance from closest cache': [], 'site spacing': [], 'n_caches': []
        }

    activations_1 = {
        'cache corr': [], 'distance': []
        }
    
    activations_2 = {
        'place-cache corr': []
        }
    
    summary_stats = {
        'identification_1': identification_1, 'identification_2': identification_2,
        'identification_3': identification_3,
        'reconstruct_1': reconstruct_1, 'reconstruct_2': reconstruct_2,
        'activations_1': activations_1, 'activations_2': activations_2
        }
    
    cache_state_idxs = np.zeros(readout.size).astype(bool)
    for c in cache_states:
         cache_state_idxs[c] = True
    noncache_state_idxs = np.logical_not(cache_state_idxs)
    dist_from_attractor, dist_counts, dist_states = \
        get_dist_from_attractor(cache_state_idxs)

    # Identification 1
    nan_idxs = np.isnan(readout)
    for threshold in np.arange(0.0, 1.0, 0.1):
        _readout = np.digitize(readout, [threshold])
        _readout[nan_idxs] = -1
        true_pos = np.sum(_readout[cache_state_idxs]==1)
        true_neg = np.sum(_readout[noncache_state_idxs]==0)
        acc = (true_pos + true_neg)/num_states
        identification_1['threshold'].append(threshold)
        identification_1['site spacing'].append(site_spacing)
        identification_1['search strength'].append(search_strength)
        identification_1['accuracy'].append(acc)
        identification_1['sensitivity'].append(true_pos/n_caches)
        identification_1['specificity'].append(true_neg/n_noncaches)
        identification_1['n_caches'].append(n_caches)

    # Identification 2
    c1 = cache_states[0]; c2 = cache_states[1];
    cache_val = min([readout[c1], readout[c2]])
    nc1 = floor((c1 + c2)/2); nc2 = ceil((c1 + c2)/2);
    noncache_val = (readout[nc1] + readout[nc2])/2
    
    identification_2['noncache val'].append(noncache_val)
    identification_2['noncache diff'].append(noncache_val-cache_val)
    identification_2['site spacing'].append(site_spacing)
    identification_2['search strength'].append(search_strength)
    identification_2['n_caches'].append(n_caches)
    
    # Identification 3
    vals = readout[cache_state_idxs]
    identification_3['site spacing'].extend([site_spacing]*n_caches)
    identification_3['search strength'].extend([search_strength]*n_caches)
    identification_3['cache'].extend(list(np.arange(n_caches)+1))
    identification_3['val'].extend(list(vals))
    identification_3['n_caches'].extend([n_caches]*n_caches)    
    
    # Reconstruction 1
    peak_locs = np.argmax(reconstruct, axis=1)
    peak_locs = (peak_locs/N_inp)*num_states
    is_cache = np.isin(peak_locs, cache_states).tolist()
    high_readout = (readout > 0.5).tolist()
    is_closest = []
    for idx in range(num_states):
        dist_from_chosen = distance(idx, peak_locs[idx], num_states)
        closest = dist_from_chosen == dist_from_attractor[idx]
        is_closest.append(closest and is_cache[idx])
    reconstruct_1['is_cache'].extend(is_cache)
    reconstruct_1['high_readout'].extend(high_readout)
    reconstruct_1['is_closest'].extend(is_closest)
    reconstruct_1['search strength'].extend([search_strength]*num_states)
    reconstruct_1['site spacing'].extend([site_spacing]*num_states)
    reconstruct_1['distance from closest cache'].extend(dist_from_attractor)
    reconstruct_1['n_caches'].extend([n_caches]*num_states)

    # Reconstruction 2
    for idx in range(num_states):
        if not is_closest[idx]: continue
        r = reconstruct[idx]/reconstruct[idx].max()
        norm_error = [np.linalg.norm(r - inputs[s]/inputs[s].max()) for s in dist_states[idx]]
        norm_error = min(norm_error)
        reconstruct_2['norm error'].append(norm_error)
        reconstruct_2['search strength'].append(search_strength)
        reconstruct_2['distance from closest cache'].append(dist_from_attractor[idx])
        reconstruct_2['site spacing'].append(site_spacing)
        reconstruct_2['n_caches'].append(n_caches)
    
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
