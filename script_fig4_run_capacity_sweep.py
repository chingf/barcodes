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

n_cpu_jobs = 40

# Re-define arguments for capacity test
N_inp = 5000
N_bar = 5000
num_states = 1000
n_seeds = 30

# Make directories
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/barcodes2/' # Local Path
elif 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/barcodes2/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/barcodes2/' # Cortex Path

exp_dir = engram_dir + 'capacity/'
os.makedirs(exp_dir, exist_ok=True)
args = []

# Set up arguments
place_decays = np.linspace(0.01, 0.25, num=10, endpoint=True)
for place_decay in place_decays:
    for resolution in np.arange(10, 110, 10):  # 210
        max_caches = int(num_states//resolution)
        for n_caches in np.arange(1, max_caches+1):
            cache_states = [int(n*resolution) for n in range(n_caches)]
            args.append([
                place_decay,
                cache_states,
                f"{place_decay:.2f}/res{resolution}/"
                f"caches{n_caches}/"
                ])

            
def cpu_parallel(args):
    job_results = Parallel(n_jobs=n_cpu_jobs)(delayed(run)(arg) for arg in args)

    
def run(arg):
    # Unpack arguments
    place_decay, cache_states, add_to_dir_path = arg
    _exp_dir = exp_dir + add_to_dir_path
    os.makedirs(_exp_dir, exist_ok=True)
    if os.path.isfile(f'{_exp_dir}results.p'):
        print('Skipping ' + _exp_dir + '...')
        return

    # Setup
    place_inputs = PlaceInputs(N_inp, num_states, decay_constant=place_decay).get_inputs()
    results = {}
    for seed in range(n_seeds):
        np.random.seed(seed)
        model = Model(N_inp, N_bar, num_states)
        _, acts, _, _ = model.run_recurrent(place_inputs)

        # Sequentially store seeds
        for cache_state in cache_states:
            preacts, acts, _, _ = model.run_recurrent(place_inputs)
            model.update(
                place_inputs[cache_state],
                acts[cache_state], preacts[cache_state]
                )
        
        # Get seed output and save to results
        _, _, outputs, _ = model.run_recall(0., place_inputs)
        _, seed_outputs = outputs
        results[seed] = seed_outputs
        
    with open(f'{_exp_dir}results.p', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    # Run script
    import time
    start = time.time()
    cpu_parallel(args)
    end = time.time()
    print(f'ELAPSED TIME: {end-start} seconds')
