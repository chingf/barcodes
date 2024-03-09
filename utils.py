import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.stats import nbinom

def relu(x, bias=0):
    return np.clip(x-bias, 0, np.inf)

def normalize(x, ax=1):
    return (x - np.mean(x, axis=ax, keepdims=True))/ (1e-8+np.std(x, axis=ax, keepdims=True))

def pairwise_correlations_centered(x):
    return np.corrcoef(x-np.mean(x, 0))

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def poiss_corr(a, b, scale=1., shift=0.):
    poiss_a = poisson(scale*(a+shift)).rvs()
    poiss_b = poisson(scale*(b+shift)).rvs()
    return np.corrcoef(poiss_a, poiss_b)[0, 1]

def nb_corr(a, b):
    nb_a = nb(a)
    nb_b = nb(b)
    return np.corrcoef(nb_a, nb_b)[0, 1]

def nb(mu, std_scaling=1.0, mu_scaling=0.75, shift=0.0):
    """ mu is a vector of firing rates. std_scaling is a scalar. """


    mu = mu*mu_scaling + 1E-8 + shift
    std = std_scaling * np.sqrt(mu)
    std += 1E-8
    n = (mu**2)/(std**2 - mu)
    p = mu/(std**2)
    nb_mu = nbinom.rvs(n, p)
    return nb_mu.astype(float)

def distance2D(a, b, arena_width):
    a_x = a // arena_width; a_y = a % arena_width;
    b_x = b // arena_width; b_y = b % arena_width;
    x_dist = max(a_x, b_x) - min(a_x, b_x)
    y_dist = max(a_y, b_y) - min(a_y, b_y)
    return np.linalg.norm([x_dist, y_dist])

def distance(a, b, maximum):
    dist = np.abs(a - b)
    dist = min(dist, np.abs(maximum-dist))
    return dist

def zero_out_invalid(reconstruct, threshold):
    reconstruct = reconstruct.copy()
    reconstruct_norm = np.linalg.norm(reconstruct, axis=1)
    reconstruct_norm /= reconstruct_norm.max()
    invalid_recall = reconstruct_norm < threshold
    reconstruct[invalid_recall] = 0
    return reconstruct

def recall_plots(
    cache_identification, narrow_recall, wide_recall,
    cache_states, recall_downsampling_idxs=None):
    
    fig, ax = plt.subplots(1, 3, figsize=(8,2))
    num_states, N_bar = cache_identification.shape
    threshold = 0.5
    if recall_downsampling_idxs is not None:
        narrow_recall = narrow_recall[:,recall_downsampling_idxs]
        wide_recall = wide_recall[:,recall_downsampling_idxs]
    
    # Identification plot
    readout = np.linalg.norm(cache_identification, axis=1)
    readout /= readout.max()
    ax[0].plot(readout)
    idxs = readout > threshold
    y = readout[idxs]
    y[y>0] = 1.05
    ax[0].scatter(np.arange(100)[idxs], y, s=1, color='red')
    ax[0].set_yticks([0, 0.50, 1.0])
    ax[0].set_ylabel('Output Norm')
    
    # Narrow recall plot
    narrow_recall = zero_out_invalid(narrow_recall, threshold)
    ax[1].imshow(narrow_recall, aspect='auto')
    ax[1].set_yticks([0, num_states//2, num_states], [0, '$\pi$', '$2\pi$'])
    
    # Wide recall plot
    wide_recall = zero_out_invalid(wide_recall, threshold)
    ax[2].imshow(wide_recall, aspect='auto')
        
    ax[1].set_ylabel('Location')
    ax[2].set_ylabel('')
    ax[2].set_yticks([])
    for _ax in [ax[0]]: # For cache identification
        xtick_loc = []; xtick_label = [];
        for i, c in enumerate(cache_states):
            xtick_loc.append(c)
            xtick_label.append(f'C{i+1}')
        _ax.set_xticks(xtick_loc)
        _ax.set_xticklabels(xtick_label, fontsize=10)
    for _ax in ax[1:]: # For recalled place fields
        xtick_loc = []; xtick_label = [];
        for i, c in enumerate(cache_states):
            if recall_downsampling_idxs is not None:
                xtick_loc.append((c/num_states)*recall_downsampling_idxs.size)
            else:
                xtick_loc.append((c/num_states)*N_bar)
            xtick_label.append(f'C{i+1}')
        _ax.set_xticks(xtick_loc)
        _ax.set_xticklabels(xtick_label, fontsize=10)
    plt.tight_layout()
    plt.show()
