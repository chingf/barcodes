import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shortuuid
import pandas as pd
from Model import Model
from PlaceInputs import PlaceInputs
from utils import *
import configs
import pickle

# Fixed parameters
N_inp = 5000
N_bar = 5000
num_states = 100
place_inputs = PlaceInputs(N_inp, num_states).get_inputs()
cache_state = 50
cache_unit = (cache_state/num_states)*N_inp
n_cpu_jobs = 20

# Collect args
n_seeds = 35
w1s = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
w2s = [1.4, 1.5, 1.6, 1.7]
args = []
for s in np.arange(n_seeds):
    for w1 in w1s:
        for w2 in w2s:
            args.append([s, w1, w2])

# Make directories
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/barcodes2/' # Local Path
elif 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/barcodes2/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/barcodes2/' # Cortex Path
exp_dir = engram_dir + 'pred_stats/'
os.makedirs(exp_dir, exist_ok=True)

def cpu_parallel():
    job_results = Parallel(n_jobs=n_cpu_jobs)(delayed(run)(arg) for arg in args)

def run(arg):
    seed, w1, w2 = arg
    np.random.seed(seed)
    results = {
        'shifts': [],
        'w1': [],
        'w2': [],
        'seed': []
    }

    # Initialize model
    model = Model(
        N_inp, N_bar, num_states,
        narrow_search_factor=0.5, wide_search_factor=1.5)
    unskewed_J = model.J_xx.copy()
    
    # Get default output
    _, place_acts, _, _ = model.run_nonrecurrent(place_inputs)
    preacts, acts, _, acts_over_time = model.run_recurrent(place_inputs)
    model.update(place_inputs[cache_state], acts[cache_state], preacts[cache_state])
    _, _, outputs, _ = model.run_recall(0., place_inputs)
    _default_readout = outputs[1].squeeze()
    _default_readout = _default_readout/_default_readout.max()
    start_idx = np.argwhere(_default_readout>0.1)[0,0]
    end_idx = np.argwhere(_default_readout>0.1)[-1,0]
    center_of_mass = (end_idx - start_idx)/2

    # Add predictive skew
    with open('fig6_pred_matrix.p', 'rb') as f:
            total_delta = pickle.load(f)
    model.J_xx = (w1)*unskewed_J.copy() + (w2)*total_delta

    # Get predictive output
    _, place_acts, _, _ = model.run_nonrecurrent(place_inputs)
    preacts, acts, _, acts_over_time = model.run_recurrent(place_inputs)
    model.update(place_inputs[cache_state], acts[cache_state], preacts[cache_state])
    _, _, outputs, _ = model.run_recall(0., place_inputs)
    _pred_readout = outputs[1].squeeze()
    _pred_readout = _pred_readout/_pred_readout.max()
    start_idx = np.argwhere(_pred_readout>0.1)[0,0]
    end_idx = np.argwhere(_pred_readout>0.1)[-1,0]
    pred_center_of_mass = (end_idx - start_idx)/2
    
    # Get diff
    results['shifts'].append(center_of_mass - pred_center_of_mass)
    results['w1'].append(w1)
    results['w2'].append(w2)
    results['seed'].append(seed)

    # Save to file
    filepath = exp_dir + shortuuid.uuid() + '.p'
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    # Run script
    import time
    start = time.time()
    cpu_parallel()
    end = time.time()
    print(f'ELAPSED TIME: {end-start} seconds')
