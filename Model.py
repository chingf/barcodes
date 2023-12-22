import numpy as np
from utils import *

class Model():
    def __init__(self,
        N_inp, N_bar, num_states,
        rec_strength=9.0, weight_bias=-10, # Initial weights
        divisive_normalization=30.0, steps=100, dt=0.1, # Dynamics
        lr=40.0, plasticity_bias = -0.7, # Learning
        narrow_search_factor=0.75, wide_search_factor=1.25,
        ):

        self.N_inp = N_inp
        self.N_bar = N_bar
        self.num_states = num_states
        self.rec_strength = rec_strength
        self.weight_bias = weight_bias
        self.divisive_normalization = divisive_normalization
        self.steps = steps
        self.dt = dt
        self.lr = lr
        self.plasticity_bias = plasticity_bias
        self.narrow_search_factor = narrow_search_factor
        self.wide_search_factor = wide_search_factor
        self.reset()

    def reset(self):
        self.J_xy = np.zeros([self.N_inp, self.N_bar])
        rand_J = np.random.randn(self.N_bar, self.N_bar)
        self.J_xx = self.rec_strength*(rand_J / np.sqrt(self.N_bar))
        self.J_xx += (self.weight_bias / self.N_bar)
        self.J_sx = np.zeros([self.N_bar])

    def run_nonrecurrent(self, inputs, n_zero_input=0):
        return self.run(inputs, n_zero_input, np.zeros(self.J_xx.shape))

    def run_recurrent(self, inputs, n_zero_input=0):
        return self.run(inputs, n_zero_input)

    def run_wide_recall(self, inputs, n_zero_input=0):
        return self.run_recall(self.wide_search_factor, inputs, n_zero_input)

    def run_narrow_recall(self, inputs, n_zero_input=0):
        return self.run_recall(self.narrow_search_factor, inputs, n_zero_input)

    def run_recall(self, search_factor, inputs, n_zero_input=0):
        return self.run(inputs+search_factor*self.J_sx, n_zero_input)

    def run(self, inputs, n_zero_input=0, J_xx=None):
        N_inp = self.N_inp; N_bar = self.N_bar; num_states = self.num_states
        divisive_normalization = self.divisive_normalization;
        steps = self.steps; dt = self.dt; 
        if J_xx is None:
            J_xx = self.J_xx

        preacts = np.zeros([num_states, N_bar])
        acts = np.zeros([num_states, N_bar])
        acts_over_time = np.zeros([steps+n_zero_input, num_states, N_bar])
        for s in range(steps):
            preacts = preacts*(1 - divisive_normalization*np.sum(acts, axis=1, keepdims=True)/N_bar * dt) + dt*np.matmul(acts, J_xx)+dt*inputs
            acts = relu(preacts)

            final_preacts = preacts.copy()
            final_acts = acts.copy()
            acts_over_time[s] = final_acts
        for s in range(n_zero_input):
            preacts = preacts*(1-divisive_normalization*np.sum(acts, axis=1, keepdims=True)/N_bar*dt) + dt*np.matmul(acts, J_xx)
            acts = relu(preacts)
            acts_over_time[steps+s] = acts.copy()
        final_output = np.matmul(final_acts, self.J_xy.transpose())
        return final_preacts, final_acts, final_output, acts_over_time

    def update(self, inputs, act, preact):
        self.J_sx += act
        self.J_xy += np.outer(inputs, act)

        act = act.reshape((1, -1))
        preact = preact.reshape((1, -1))
        delta_J = np.matmul(act.transpose(), preact)
        delta_J += np.matmul(np.ones_like(act.transpose())*self.plasticity_bias, act)
        self.J_xx += self.lr * delta_J / self.N_bar
   
