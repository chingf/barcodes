import sys
import pickle
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from Model import Model
from PlaceInputs import PlaceInputs
from utils import *
import configs

from Fig3_run_resolution_sweep import * # Bulk of script is here

# Re-define arguments for capacity test
exp_dir = engram_dir + 'capacity/' + exp + '/' + model_type + '/'
os.makedirs(exp_dir, exist_ok=True)
args = []
param_sweep_search_strengths = [0., 0.2, 0.4, 0.6, 0.8]
if exp == 'narrow_search_factor':
    exp_params = []
    for v in param_sweep_search_strengths:
        exp_params.append({exp: v})
for exp_param in exp_params:
    for seed in range(n_seeds):
        for spacing in [15]:
            for n_caches in np.arange(1,7):
                params = model_params.copy()
                for key in exp_param.keys():
                    params[key] = exp_param[key]
                cache_states = [int(n*spacing) for n in range(n_caches)]
                args.append([
                    params, seed, cache_states,
                    f"{exp_param[exp]:.1f}/spacing{spacing}/"
                    f"caches{n_caches}/seed{seed}/"
                    ])

if __name__ == '__main__':
    # Run script
    import time
    for arg in args:
        start = time.time()
        run(arg)
        end = time.time()
        print(f'ELAPSED TIME: {end-start} seconds')
