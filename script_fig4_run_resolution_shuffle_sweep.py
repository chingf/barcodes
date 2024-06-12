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

from script_fig4_run_resolution_sweep import * # Bulk of script is here


# Redefine experiment dir
exp_dir = engram_dir + 'resolution_shuffle/' + exp + '/' + model_type + '/'
os.makedirs(exp_dir, exist_ok=True)

# Set up arguments, with shuffling
np.random.seed(0)
args = []
for exp_param in exp_params:
    for seed in range(n_seeds):
        for resolution in [3, 5, 10, 15, 20, 25, 30]:
            params = model_params.copy()
            for key in exp_param.keys():
                params[key] = exp_param[key]
            cache_states = [0, resolution, 66]
            while cache_states == [0, resolution, 66]:
                np.random.shuffle(cache_states)
            print(cache_states)
            args.append([
                params, seed, cache_states,
                f'{exp_param[exp]:.1f}/res{resolution}/seed{seed}/', exp_dir
                ])


if __name__ == '__main__':
    # Run script
    import time
    start = time.time()
    cpu_parallel(args)
    end = time.time()
    print(f'ELAPSED TIME: {end-start} seconds')

    #for arg in args:
    #    start = time.time()
    #    run(arg)
    #    end = time.time()
    #    print(f'ELAPSED TIME: {end-start} seconds')
