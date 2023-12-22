import numpy as np
from scipy.stats import poisson

def relu(x, bias=0):
    return np.clip(x-bias, 0, np.inf)

def normalize(x, ax=1):
    return (x - np.mean(x, axis=ax, keepdims=True))/ (1e-8+np.std(x, axis=ax, keepdims=True))

def pairwise_correlations_centered(x):
    return np.corrcoef(x-np.mean(x, 0))

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def poiss_corr(a, b):
    scale = 1.
    poiss_a = poisson(scale*a).rvs()
    poiss_b = poisson(scale*b).rvs()
    return np.corrcoef(poiss_a, poiss_b)[0, 1]

def distance(a, b, maximum):
    dist = np.abs(a - b)
    dist = min(dist, np.abs(maximum-dist))
    return dist
