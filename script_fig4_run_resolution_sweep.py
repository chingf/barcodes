import sys
import pickle
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from Model import Model
from PlaceInputs import PlaceInputs, PlaceInputsExp
from utils import *
import configs

n_cpu_jobs = 40

# Determine experiment
exp = sys.argv[1]
model_type = sys.argv[2]
n_seeds = 20
args = []

exp_params = []
param_sweep_search_strengths = [0., 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2.0]
if exp == 'narrow_search_factor': # Just vary search strength
    for v in param_sweep_search_strengths:
        exp_params.append({exp: v})
elif exp == 'weight_var': # Strength of recurrent dynamics
    for v in [0, 3, 5, 7, 9, 11, 13, 15]:
        exp_params.append({exp: v})
elif exp == 'weight_bias': # Offset in initial weights of J_xx
    for v in [-200, -100, -50, -10, 0, 10]:
        exp_params.append({exp: v})
elif exp == 'divisive_normalization': # Strength of dynamics normalization
    for v in [15, 30, 45, 60]:
        exp_params.append({exp: v})
elif exp == 'plasticity_bias': # Offset in plasticity update
    for v in [0, -0.1, -0.2, -0.3, -0.4, -0.6, -0.7, -0.8, -0.9, -1.0]:
        if v >= -0.2: 
            lr = 10
        elif v >= -0.4:
            lr = 20
        elif v >= -0.5:
            lr = 30
        else:
            lr = 50
        exp_params.append({exp: v, 'lr': lr})

# Fixed parameters
N_inp = 5000
N_bar = 5000
num_states = 100
PlaceClass = PlaceInputs
if model_type == 'default':
    model_params = {}
    place_input_params = {}
elif model_type == 'prediction':
    model_params = {'plasticity_bias': -0.45, 'add_pred_skew': True}
    place_input_params = {}
elif model_type == 'big':
    model_params = {}
    place_input_params = {}
    N_inp *= 3; N_bar *= 3
    n_seeds = 10
elif model_type == 'barcode_ablation':
    model_params = {'rec_strength': 0.0, 'weight_bias': 0.}
    if exp == 'rec_strength':
        raise ValueError('Incompatable parameter sweep + model setting')
    place_input_params = {}
elif model_type == 'lr_ablation':
    model_params = {'lr': 0.0}
    if exp == 'plasticity_bias':
        raise ValueError('Incompatable parameter sweep + model setting')
    place_input_params = {}
elif model_type == 'place_field_ablation':
    model_params = {}
    place_input_params = {'decay_constant': 0.0001}
elif model_type == 'gaussian':
    model_params = {
        'lr': 75, 'plasticity_bias': -0.32, 'rec_strength': 7.0,
        'weight_bias': -40, 'divisive_normalization': 30.0, 'seed_strength_cache': 1.5}
    PlaceClass = PlaceInputsExp
    place_input_params = {}
    

# Set up arguments
for exp_param in exp_params:
    for seed in range(n_seeds):
        for resolution in [3, 5, 10, 15, 20, 25, 30]:
            params = model_params.copy()
            for key in exp_param.keys():
                params[key] = exp_param[key]
            cache_states = [0, resolution, 66]
            args.append([
                params, seed, cache_states,
                f'{exp_param[exp]:.1f}/res{resolution}/seed{seed}/'
                ])

# Make directories
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/barcodes2/' # Local Path
elif 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/barcodes2/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/barcodes2/' # Cortex Path
exp_dir = engram_dir + 'resolution/' + exp + '/' + model_type + '/'
os.makedirs(exp_dir, exist_ok=True)

def cpu_parallel():
    job_results = Parallel(n_jobs=n_cpu_jobs)(delayed(run)(arg) for arg in args)

def run(arg):
    # Unpack arguments
    params, seed, cache_states, add_to_dir_path  = arg
    _exp_dir = exp_dir + add_to_dir_path
    os.makedirs(_exp_dir, exist_ok=True)
    if os.path.isfile(f'{_exp_dir}results.p'):
        print('Skipping ' + _exp_dir + '...')
        return

    # Setup
    place_inputs = PlaceClass(N_inp, num_states, **place_input_params).get_inputs()
    model = Model(N_inp, N_bar, num_states, **params)
    _, acts, _, _ = model.run_recurrent(place_inputs)
    results = {'initial_acts': acts,}

    # Run task
    for cache_num, cache_state in enumerate(cache_states):
        
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))
        preacts, acts, _, _ = model.run_recurrent(place_inputs)
        model.update(place_inputs[cache_state], acts[cache_state], preacts[cache_state])
        results['J_xx'] = model.J_xx
       
        _, acts, outputs, _ = model.run_recall(0., place_inputs)
        seed_reconstruct = outputs[1]
        ax[0].set_title("Barcode Activity Correlation,\nfantasy OFF")
        ax[0].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')

        _, wide_acts, outputs, _ = model.run_wide_recall(place_inputs)
        wide_reconstruct = outputs[0]
        if model_type == 'gaussian':
            wide_reconstruct = wide_reconstruct@place_inputs.transpose()
        ax[1].set_title("Barcode Activity Correlation,\nfantasy (wide search)")
        ax[1].imshow(pairwise_correlations_centered(wide_acts), vmin=0, vmax=1, aspect='auto')
        ax[2].set_xlabel("Neuron")
        ax[2].set_ylabel("Location")
        ax[2].imshow(wide_reconstruct, aspect='auto')
        ax[2].set_title("Place field reconstruction,\nfantasy (wide search)")

        _, narrow_acts, outputs, _ = model.run_narrow_recall(place_inputs)
        narrow_reconstruct = outputs[0]
        if model_type == 'gaussian':
            narrow_reconstruct = narrow_reconstruct@place_inputs.transpose()
        ax[3].set_title("Barcode Activity Correlation,\nfantasy (narrow search)")
        ax[3].imshow(pairwise_correlations_centered(narrow_acts), vmin=0, vmax=1, aspect='auto')
        ax[4].set_xlabel("Neuron")
        ax[4].set_ylabel("Location")
        ax[4].imshow(narrow_reconstruct, aspect='auto')
        ax[4].set_title("Place field reconstruction,\nfantasy (narrow search)")
        
        for _ax in [ax[0], ax[1], ax[3]]:
            _ax.set_xlabel('Location')
            _ax.set_ylabel('Location')
            
        plt.savefig(_exp_dir + f'cache{cache_num}.png', dpi=300)

        if exp == 'narrow_search_factor':
            results['wide_acts'] = wide_acts
            results['wide_reconstruct'] = wide_reconstruct
            results['narrow_acts'] = narrow_acts
            results['narrow_reconstruct'] = narrow_reconstruct
            results['seed_reconstruct'] = seed_reconstruct

    if exp != 'narrow_search_factor':
        for s in param_sweep_search_strengths: 
            _, acts, outputs, _ = model.run_recall(s, place_inputs)
            reconstruct, seed_reconstruct = outputs
            if model_type == 'gaussian':
                reconstruct = reconstruct@place_inputs.transpose()
            results[f'{s:.2f}_acts'] = acts
            results[f'{s:.2f}_reconstruct'] = reconstruct
            results[f'{s:.2f}_seed_reconstruct'] = seed_reconstruct

    # Save data structures
    plt.close('all')
    with open(f'{_exp_dir}results.p', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    # Run script
    import time
    start = time.time()
    cpu_parallel()
    end = time.time()
    print(f'ELAPSED TIME: {end-start} seconds')

    #for arg in args:
    #    start = time.time()
    #    run(arg)
    #    end = time.time()
    #    print(f'ELAPSED TIME: {end-start} seconds')
