import numpy as np
from utils import *

class Model():
    def __init__(self,
        N_inp, N_bar, num_states,
        rec_strength=7.0, weight_bias=-40, # Initial weights
        divisive_normalization=20.0, steps=100, seed_steps = 5, dt=0.1, # Dynamics
        lr=50.0, plasticity_bias = -0.35, # Learning
        narrow_search_factor=0.0, wide_search_factor=0.7, seed_strength_cache=3.0,
        forget_readout_lr=0.25, forget_lr=3.5, forget_plasticity_bias=-2.25
        ):

        self.N_inp = N_inp
        self.N_bar = N_bar
        self.num_states = num_states
        self.rec_strength = rec_strength
        self.weight_bias = weight_bias
        self.divisive_normalization = divisive_normalization
        self.steps = steps
        self.seed_steps = seed_steps
        self.dt = dt
        self.lr = lr
        self.plasticity_bias = plasticity_bias
        self.narrow_search_factor = narrow_search_factor
        self.wide_search_factor = wide_search_factor
        self.seed_strength_cache = seed_strength_cache
        self.forget_readout_lr = forget_readout_lr
        self.forget_lr = forget_lr
        self.forget_plasticity_bias = forget_plasticity_bias
        self.reset()

    def reset(self):
        self.J_xy = np.zeros([self.N_inp, self.N_bar])
        rand_J = np.random.randn(self.N_bar, self.N_bar)
        self.J_xx = self.rec_strength*(rand_J / np.sqrt(self.N_bar))
        self.J_xx += (self.weight_bias / self.N_bar)
        self.J_xx_orig = np.copy(self.J_xx)
        self.J_sx = np.random.randn(self.N_bar)

    def run_nonrecurrent(self, inputs, n_zero_input=0):
        return self.run(inputs, n_zero_input, np.zeros(self.J_xx.shape), seed_steps=0)

    def run_recurrent(self, inputs, n_zero_input=0):
        return self.run(inputs, n_zero_input)

    def run_wide_recall(self, inputs, n_zero_input=0):
        return self.run_recall(self.wide_search_factor, inputs, n_zero_input)

    def run_narrow_recall(self, inputs, n_zero_input=0):
        return self.run_recall(self.narrow_search_factor, inputs, n_zero_input)

    def run_recall(self, search_factor, inputs, n_zero_input=0):
        return self.run(inputs+search_factor*self.J_sx, n_zero_input, seed_steps=0)

    def run_seed_only(self, seed_steps=5, search_factor=1.):
        N_inp = self.N_inp; N_bar = self.N_bar; num_states = self.num_states
        divisive_normalization = self.divisive_normalization; J_xx = self.J_xx
        dt = self.dt

        preacts = np.zeros([num_states, N_bar])
        acts = np.zeros([num_states, N_bar])
        acts_over_time = np.zeros([seed_steps, num_states, N_bar])
        for s in range(seed_steps):
            preacts = preacts*(1 - divisive_normalization*np.sum(acts, axis=1, keepdims=True)/N_bar * dt) + dt*np.matmul(acts, J_xx)+dt*(self.J_sx*search_factor)
            acts = relu(preacts)
            acts_over_time[s] = acts.copy()
        output = np.matmul(acts, self.J_xy.transpose())
        return preacts, acts, output, acts_over_time

    def run(self, inputs, n_zero_input=0, J_xx=None, seed_steps = None):
        if seed_steps is None:
            seed_steps=self.seed_steps
        N_inp = self.N_inp; N_bar = self.N_bar; num_states = self.num_states
        divisive_normalization = self.divisive_normalization;
        steps = self.steps; dt = self.dt; 
        seed_strength_cache=self.seed_strength_cache;
        if J_xx is None:
            J_xx = self.J_xx

        preacts = np.zeros([num_states, N_bar])
        acts = np.zeros([num_states, N_bar])
        acts_over_time = np.zeros([steps+seed_steps+n_zero_input, num_states, N_bar])
        for s in range(steps):
            preacts = preacts*(1 - divisive_normalization*np.sum(acts, axis=1, keepdims=True)/N_bar * dt) + dt*np.matmul(acts, J_xx)+dt*inputs
            acts = relu(preacts)
            acts_over_time[s] = acts.copy()
            
        for s in range(seed_steps):
            preacts = preacts*(1 - divisive_normalization*np.sum(acts, axis=1, keepdims=True)/N_bar * dt) + dt*np.matmul(acts, J_xx)+dt*(inputs+self.J_sx*seed_strength_cache)
            acts = relu(preacts)
            acts_over_time[steps+s] = acts.copy()

        final_preacts = preacts.copy()
        final_acts = acts.copy()
        for s in range(n_zero_input):
            preacts = preacts*(1-divisive_normalization*np.sum(acts, axis=1, keepdims=True)/N_bar*dt) + dt*np.matmul(acts, J_xx)
            acts = relu(preacts)
            acts_over_time[steps+seed_steps+s] = acts.copy()
        final_output = np.matmul(final_acts, self.J_xy.transpose())
        return final_preacts, final_acts, final_output, acts_over_time

    def update(self, inputs, act, preact):
        self.J_xy += np.outer(inputs, act)
        act = act.reshape((1, -1))
        preact = preact.reshape((1, -1))
        delta_J = np.matmul(act.transpose()+self.plasticity_bias, act)
        self.J_xx += self.lr * delta_J / self.N_bar
   
    def reverse_update(self, act):
        recon = np.dot(self.J_xy, act)
        self.J_xy -= self.forget_readout_lr * np.outer(recon, act) / self.N_bar
        act = act.reshape((1, -1))
        delta_J = np.matmul((act.transpose() + self.forget_plasticity_bias), act)
        self.J_xx -= self.forget_lr * delta_J / self.N_bar

