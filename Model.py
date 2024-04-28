import numpy as np
from utils import *
import warnings

class Model():
    def __init__(self,
        N_inp, N_bar, num_states,
        weight_var=7.0, weight_bias=-40, gaussian_J_ix=False, # Initial weights
        divisive_normalization=20.0, steps=100, seed_steps = 5, dt=0.1, # Dynamics
        lr=40.0, plasticity_bias = -0.38, # Learning
        narrow_search_factor=0.0, wide_search_factor=0.7, seed_strength_cache=3.0,
        forget_readout_lr=0.25, forget_lr=3.5, forget_plasticity_bias=-2.25,
        add_pred_skew=False
        ):

        self.N_inp = N_inp
        self.N_bar = N_bar
        if N_inp != N_bar and not gaussian_J_ix:
            warnings.warn('N_inp != N_bar ; J_ix weights will be Gaussian.')
            gaussian_J_ix = True
        self.gaussian_J_ix = gaussian_J_ix
        self.num_states = num_states
        self.weight_var = weight_var
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
        self.add_pred_skew = add_pred_skew
        self.reset()

    def reset(self):
        self.J_xy = np.zeros([self.N_inp, self.N_bar])
        self.J_xs = np.zeros([1, self.N_bar])
        rand_J = np.random.randn(self.N_bar, self.N_bar)
        self.J_xx = self.weight_var*(rand_J / np.sqrt(self.N_bar))
        self.J_xx += (self.weight_bias / self.N_bar)
        self.J_xx_orig = np.copy(self.J_xx)
        self.J_sx = np.random.randn(self.N_bar)

        if self.gaussian_J_ix:
            self.J_ix = np.random.randn(self.N_inp, self.N_bar) / np.sqrt(self.N_inp)
        else:
            self.J_ix = np.eye(self.N_inp)

        if self.add_pred_skew:
            identity = np.eye(self.N_bar)
            n_shifts = int(self.N_bar/10.)
            shift_offset = int(self.N_bar/self.num_states)
            gamma = 0.99
            for s in range(1, n_shifts):
                shifted = np.roll(identity, shift=-(s+shift_offset), axis=0)
                delta = (gamma**s)*0.04*shifted
                self.J_xx += delta

    def run_nonrecurrent(self, inputs, n_zero_input=0):
        return self.run(inputs, n_zero_input, np.zeros(self.J_xx.shape), seed_steps=0)

    def run_recurrent(self, inputs, n_zero_input=0, rec_strength=1., seed_steps=None):
        return self.run(inputs, n_zero_input, rec_strength=rec_strength, seed_steps=seed_steps)

    def run_wide_recall(self, inputs, n_zero_input=0):
        return self.run_recall(self.wide_search_factor, inputs, n_zero_input)

    def run_narrow_recall(self, inputs, n_zero_input=0):
        return self.run_recall(self.narrow_search_factor, inputs, n_zero_input)

    def run_recall(self, search_factor, inputs, n_zero_input=0, rec_strength=1.):
        return self.run(
            inputs, n_zero_input, rec_strength=rec_strength,
            seed_steps=0, search_factor=search_factor)

    def run(
        self, raw_inputs, n_zero_input=0,
        J_xx=None, rec_strength=1.,
        seed_steps = None, search_factor=0.0):
        
        if seed_steps is None:
            seed_steps=self.seed_steps
            
        inputs = raw_inputs @ self.J_ix + search_factor*self.J_sx
        N_inp = self.N_inp; N_bar = self.N_bar; num_states = self.num_states
        divisive_normalization = self.divisive_normalization;
        steps = self.steps; dt = self.dt; 
        seed_strength_cache=self.seed_strength_cache;
        if J_xx is None:
            J_xx = self.J_xx
        J_xx = J_xx * rec_strength

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
        final_output = np.matmul(final_acts, self.J_xy.transpose()) # place field
        final_output_s = np.matmul(final_acts, self.J_xs.transpose()) # seed
        final_outputs = (final_output, final_output_s)
        return final_preacts, final_acts, final_outputs, acts_over_time

    def update(self, inputs, act, preact):
        self.J_xy += np.outer(inputs, act)
        self.J_xs += np.outer(np.ones((1,1)), act)
        act = act.reshape((1, -1))
        preact = preact.reshape((1, -1))
        delta_J = np.matmul(act.transpose()+self.plasticity_bias, act)
        self.J_xx += self.lr * delta_J / self.N_bar
   
    def reverse_update(self, act):
        """ deprecated. used during forgetting experiments. """
        recon = np.dot(self.J_xy, act)
        self.J_xy -= self.forget_readout_lr * np.outer(recon, act) / self.N_bar
        act = act.reshape((1, -1))
        preact = preact.reshape((1, -1))
        delta_J = np.matmul((act.transpose() + self.forget_plasticity_bias), act)
        self.J_xx -= self.forget_lr * delta_J / self.N_bar

