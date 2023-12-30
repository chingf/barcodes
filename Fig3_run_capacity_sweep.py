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

# Determine experiment
n_seeds = 5
exp = 'default'
params = {}

# Fixed parameters
N_inp = 2000
N_bar = 2000
num_states = 100

# Arguments are unpacked-- useful if parallelizing 
args = []
for n_caches in np.arange(1, 20):
    _cache_states = np.linspace(0, num_states, n_caches, endpoint=False).astype(int)
    for seed in range(n_seeds):
        args.append([_cache_states, params, seed, f'{n_caches}_caches/seed{seed}/'])

# Make directories
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/barcodes/' # Local Path
elif 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/barcodes/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/barcodes/' # Cortex Path
exp_dir = engram_dir + 'capacity/' + exp + '/'
os.makedirs(exp_dir, exist_ok=True)
n_cpu_jobs = 5

def cpu_parallel():
    job_results = Parallel(n_jobs=n_cpu_jobs)(delayed(run)(arg) for arg in args)

def run(arg):
    # Unpack arguments
    cache_states, params, seed, add_to_dir_path = arg
    _exp_dir = exp_dir + add_to_dir_path
    os.makedirs(_exp_dir, exist_ok=True)
    if os.path.isfile(f'{_exp_dir}results.p'):
        print('Skipping ' + _exp_dir + '...')
        return

    # Setup
    place_inputs = PlaceInputs(N_inp, num_states).get_inputs()
    model = Model(N_inp, N_bar, num_states, **params)
    s_vals = [0, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.]
    results = {}
    for s_val in s_vals:
        results[f's{s_val:.2f}_acts'] = None
        results[f's{s_val:.2f}_reconstruct'] = None

    # Initial plots
    _, input_acts, _, _ = model.run_nonrecurrent(place_inputs)
    plt.figure()
    plt.imshow(input_acts, vmin=0, vmax=2, aspect='auto')
    plt.xlabel("neuron")
    plt.ylabel("location")
    plt.colorbar()
    plt.title("barcode activity (recurrent mode)")
    plt.savefig(_exp_dir + 'init_barcode_activity.png', dpi=300)
    
    plt.figure()
    plt.imshow(pairwise_correlations_centered(input_acts), vmin=0, vmax=1)
    plt.colorbar()
    plt.title("pairwise correlations of barcode activity (recurrent mode)")
    plt.savefig(_exp_dir + 'init_barcode_corr.png', dpi=300)

    _, acts, _, _ = model.run_nonrecurrent(place_inputs)   
    acts_normalized = normalize(acts)
    inputs_normalized = normalize(input_acts)
    corrs = [np.corrcoef(acts_normalized[i], inputs_normalized[i])[0, 1] for i in range(num_states)]
    plt.figure()
    plt.plot(corrs)
    plt.title("Correlations between place and barcode activity at each state")
    plt.xlabel("Location")
    plt.savefig(_exp_dir + 'init_barcode_place_corr.png', dpi=300)

    # Run task
    for cache_num, cache_state in enumerate(cache_states):
        
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))
        preacts, acts, _, _ = model.run_recurrent(place_inputs)
        model.update(place_inputs[cache_state], acts[cache_state], preacts[cache_state])

        for s_val in s_vals:
            _, acts, reconstruct, _ = model.run_recall(s_val, place_inputs)
            results[f's{s_val:.2f}_acts'] = acts
            results[f's{s_val:.2f}_reconstruct'] = reconstruct
       
        acts = results['s0.00_acts']
        ax[0].set_title("Barcode Activity Correlation,\nfantasy OFF")
        ax[0].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')

        acts = results['s0.75_acts']
        reconstruct = results['s0.75_reconstruct']
        ax[1].set_title("Barcode Activity Correlation,\nfantasy (wide search)")
        ax[1].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')
        ax[2].set_xlabel("Neuron")
        ax[2].set_ylabel("Location")
        ax[2].imshow(reconstruct, aspect='auto')
        ax[2].set_title("Place field reconstruction,\nfantasy (wide search)")

        acts = results['s1.25_acts']
        reconstruct = results['s1.25_reconstruct']
        ax[3].set_title("Barcode Activity Correlation,\nfantasy (narrow search)")
        ax[3].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')
        ax[4].set_xlabel("Neuron")
        ax[4].set_ylabel("Location")
        ax[4].imshow(reconstruct, aspect='auto')
        ax[4].set_title("Place field reconstruction,\nfantasy (narrow search)")
        
        for _ax in [ax[0], ax[1], ax[3]]:
            _ax.set_xlabel('Location')
            _ax.set_ylabel('Location')
            
        plt.savefig(_exp_dir + f'cache{cache_num}.png', dpi=300)

    # Save data structures
    plt.close('all')
    with open(f'{_exp_dir}results.p', 'wb') as f:
        pickle.dump(results, f)

# Run script
import time
for arg in args:
    start = time.time()
    run(arg)
    end = time.time()
    print(f'ELAPSED TIME: {end-start} seconds')
