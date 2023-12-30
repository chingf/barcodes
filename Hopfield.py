import numpy as np
from utils import *

class Model():
    def __init__(self,
        N_inp, num_states,
        steps=100, dt=0.1, # Dynamics
        lr=1.0, # Learning
        ):

        self.N_inp = N_inp
        self.num_states = num_states
        self.steps = steps
        self.dt = dt
        self.lr = lr
        self.reset()

    def reset(self):
        self.J_xx = np.zeros((self.N_inp, self.N_inp))
        self.n_patterns = 0

    def run(self, inputs, n_zero_input=0, J_xx=None):
        N_inp = self.N_inp; num_states = self.num_states
        steps = self.steps; dt = self.dt; 
        if J_xx is None:
            J_xx = self.J_xx

        preacts = np.zeros([num_states, N_inp])
        acts = np.zeros([num_states, N_inp])
        acts_over_time = np.zeros([steps+n_zero_input, num_states, N_inp])
        for s in range(steps):
            preacts = (1 - dt)*preacts + dt*(np.matmul(acts, J_xx) + inputs)
            acts = preacts #np.tanh(preacts)
            final_preacts = preacts.copy()
            final_acts = acts.copy()
            acts_over_time[s] = final_acts
        return final_preacts, final_acts, acts_over_time

    def update(self, pattern):
        self.n_patterns += 1
        delta_J = np.outer(pattern, pattern)
        self.J_xx += self.lr * delta_J
        np.fill_diagonal(self.J_xx, 0)
   
