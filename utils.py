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

def nb(mu, std_scaling=1.0, mu_scaling=0.5, shift=0.02):
    """ mu is a vector of firing rates. std_scaling is a scalar. """


    mu = mu*mu_scaling + 1E-8 + shift
    std = std_scaling * np.sqrt(mu)
    std += 1E-8
    n = (mu**2)/(std**2 - mu)
    p = mu/(std**2)
    nb_mu = nbinom.rvs(n, p)
    return nb_mu.astype(float)

def distance(a, b, maximum):
    dist = np.abs(a - b)
    dist = min(dist, np.abs(maximum-dist))
    return dist

threshold=0.5

def zero_out_invalid(reconstruct, threshold):
    reconstruct = reconstruct.copy()
    reconstruct_norm = np.linalg.norm(reconstruct, axis=1)
    reconstruct_norm /= reconstruct_norm.max()
    invalid_recall = reconstruct_norm < threshold
    reconstruct[invalid_recall] = 0
    return reconstruct

def recall_plots(
    cache_identification, narrow_recall, wide_recall,
    cache_states):
    
    fig, ax = plt.subplots(1, 3, figsize=(8,2))
    num_states, N_bar = cache_identification.shape
    threshold = 0.5
    
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
    for _ax in ax[1:]: # For recalled place fields
        xtick_loc = []; xtick_label = [];
        for i, c in enumerate(cache_states):
            xtick_loc.append((c/num_states)*N_bar)
            xtick_label.append(f'C{i+1}')
    try: # Only compatible with updated version of Matplotlib
        ax[0].set_xticks(xtick_loc, xtick_label, rotation=45, color='red', fontsize=10)
        ax[1].set_xticks(xtick_loc, xtick_label, rotation=45, color='red', fontsize=10)
        ax[2].set_xticks(xtick_loc, xtick_label, rotation=45, color='red', fontsize=10)
    except:
        pass
    plt.tight_layout()
    plt.show()