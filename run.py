import sys
import pickle
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Determine experiment
exp = sys.argv[1]
n_seeds = 10
default_eta = 10; default_beta = 0.6; default_alpha = 2.0;
if exp == 'default':
    args = []
    for seed in range(n_seeds):
        params = [default_eta, default_beta, default_alpha]
        args.append([params, seed, f'seed{seed}/'])
elif exp == 'eta': # Hebbian learning rate
    args = []
    for eta in [10**i for i in range(-3,4)]:
        for seed in range(n_seeds):
            params = [eta, default_beta, default_alpha]
            args.append([params, seed, f'{eta}/seed{seed}/'])
elif exp == 'beta': # Anti-Hebbian bias
    args = []
    for beta in [0.0, 0.2, 0.4, 0.6, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        for seed in range(n_seeds):
            params = [default_eta, beta, default_alpha]
            args.append([params, seed, f'{beta}/seed{seed}/'])
elif exp == 'alpha': # Recurrent strength
    args = []
    for alpha in [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        for seed in range(n_seeds):
            params = [default_eta, default_beta, alpha]
            args.append([params, seed, f'{alpha}/seed{seed}/'])

# Make directories
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/barcodes/' # Local Path
elif 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/barcodes/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/barcodes/' # Cortex Path
exp_dir = engram_dir + exp + '/'
os.makedirs(exp_dir, exist_ok=True)
n_cpu_jobs = 25

# Fixed parameters
N_inp = 5000
N_bar = 5000
num_states = 100
decay_constant = 0.2
steps = 100
dt = 0.1
cache_states = [20, 30, 70]
input_strength = 1.0
b = 1.0

def cpu_parallel():
    job_results = Parallel(n_jobs=n_cpu_jobs)(delayed(run)(arg) for arg in args)

def run(arg):
    # Unpack arguments
    params, seed, add_to_dir_path  = arg
    lr = params[0] # ETA
    plasticity_bias = -1 * params[1] # BETA
    rec_strength = params[2] # ALPHA
    _exp_dir = exp_dir + add_to_dir_path
    os.makedirs(_exp_dir, exist_ok=True)
    if os.path.isfile(f'{_exp_dir}results.p'):
        print('Skipping ' + _exp_dir + '...')
        return

    # Setup
    inputs = np.zeros([num_states, N_inp])
    for s in range(num_states):
        peak = int(s / float(num_states) * N_inp)
        for n in range(N_inp):
            dist = distance(n, peak, N_inp)
            inputs[s, n] = np.exp(-(dist/(N_inp*decay_constant)))
    W_reconstruct = np.zeros([N_inp, N_bar])
    rand_J = np.random.randn(N_bar, N_bar)
    W_rec = rec_strength*(rand_J / np.sqrt(N_bar))
    W_fantasy = np.zeros([N_bar])
    results = {
        'narrow_acts': None, 'narrow_reconstruct': None,
        'wide_acts': None, 'wide_reconstruct': None,
        'W_rec': None, 'fantasy_off_acts': None
        }

    # Initial plots
    preacts, acts = run_dynamics(W_rec, input_strength*inputs, b=b)
    plt.figure()
    plt.imshow(acts, vmin=0, vmax=2, aspect='auto')
    plt.xlabel("neuron")
    plt.ylabel("location")
    plt.colorbar()
    plt.title("barcode activity (recurrent mode)")
    plt.savefig(_exp_dir + 'init_barcode_activity.png', dpi=300)
    
    plt.figure()
    plt.imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1)
    plt.colorbar()
    plt.title("pairwise correlations of barcode activity (recurrent mode)")
    plt.savefig(_exp_dir + 'init_barcode_corr.png', dpi=300)
    
    acts_normalized = normalize(acts)
    inputs_normalized = normalize(inputs)
    corrs = [np.corrcoef(acts_normalized[i], inputs_normalized[i])[0, 1] for i in range(num_states)]
    plt.figure()
    plt.plot(corrs)
    plt.title("Correlations between place and barcode activity at each state")
    plt.xlabel("Location")
    plt.savefig(_exp_dir + 'init_barcode_place_corr.png', dpi=300)

    narrow_search_factor = 0.05
    wide_search_factor = 0.2
    for cache_num, cache_state in enumerate(cache_states):
        
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))
        preacts, acts = run_dynamics(W_rec, input_strength*inputs, b=b)
            
        W_fantasy += acts[cache_state]
        act = acts[cache_state:cache_state+1]
        preact = preacts[cache_state:cache_state+1]
        delta_W = np.matmul(act.transpose(), act) + np.matmul(np.ones_like(act.transpose())*plasticity_bias, act)
        W_rec += lr * delta_W / N_bar
        W_reconstruct += inputs[cache_state].reshape(-1, 1) @ acts[cache_state].reshape(1, -1)
        results['W_rec'] = W_rec
        
        preacts, acts = run_dynamics(W_rec, input_strength*inputs, b=b) 
        results['fantasy_off_acts'] = acts
        ax[0].set_title("Barcode Activity Correlation,\nfantasy OFF")
        ax[0].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')
    
        preacts, acts = run_dynamics(W_rec, input_strength*inputs+wide_search_factor*W_fantasy, b=b) 
        results['wide_acts'] = acts
        ax[1].set_title("Barcode Activity Correlation,\nfantasy (wide search)")
        ax[1].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')
        
        reconstruct = np.matmul(acts, W_reconstruct.transpose())
        results['wide_reconstruct'] = reconstruct
        ax[2].set_xlabel("Neuron")
        ax[2].set_ylabel("Location")
        ax[2].imshow(reconstruct, aspect='auto')
        ax[2].set_title("Place field reconstruction,\nfantasy (wide search)")
        
        preacts, acts = run_dynamics(W_rec, input_strength*inputs+narrow_search_factor*W_fantasy, b=b) 
        results['narrow_acts'] = acts
        ax[3].set_title("Barcode Activity Correlation,\nfantasy (narrow search)")
        ax[3].imshow(pairwise_correlations_centered(acts), vmin=0, vmax=1, aspect='auto')
        
        reconstruct = np.matmul(acts, W_reconstruct.transpose())
        results['narrow_reconstruct'] = reconstruct
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

### Helper functions
def relu(x, b=0):
    return np.clip(x-b, 0, np.inf)

def normalize(x):
    return (x -np.mean(x, axis=1, keepdims=True))/ (1e-8+np.std(x, axis=1, keepdims=True))

def run_dynamics(W, inputs,  dt=0.1, b=0.0):
    preacts = np.zeros([num_states, N_bar])
    acts = np.zeros([num_states, N_bar])
    for s in range(steps):
        preacts = preacts*(1-dt) + dt*(np.matmul(acts, W))+dt*inputs
        preacts = normalize(preacts)
        acts = relu(preacts, b=b)
    return normalize(preacts), acts

def pairwise_correlations_centered(x):
    return np.corrcoef(x-np.mean(x, 0))

def distance(a, b, maximum):
    dist = np.abs(a - b)
    dist = min(dist, np.abs(maximum-dist))
    return dist

# Run script
import time
for arg in args:
    start = time.time()
    run(arg)
    end = time.time()
    print(f'ELAPSED TIME: {end-start} seconds')
