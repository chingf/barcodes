import numpy as np
from utils import *

class PlaceInputs():
    def __init__(self, N_inp, num_states, decay_constant=0.2):
        self.N_inp = N_inp
        self.num_states = num_states
        self.decay_constant = decay_constant

        inputs = np.zeros([num_states, N_inp])
        for s in range(num_states):
            peak = int(s / float(num_states) * N_inp)
            for n in range(N_inp):
                dist = distance(n, peak, N_inp)
                inputs[s, n] = np.exp(-(dist/(N_inp*decay_constant)))
        if N_inp > 5000:
            offset = (N_inp - 5000)//2
            mean = np.mean(inputs[:,offset:-offset], axis=1, keepdims=True)
            std = np.std(inputs[:,offset:-offset], axis=1, keepdims=True)
        else:
            mean = np.mean(inputs, axis=1, keepdims=True)
            std = np.std(inputs, axis=1, keepdims=True)
        inputs = inputs - mean
        inputs = inputs / std
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

class PlaceInputs2D():
    def __init__(self, N_inp, num_states, decay_constant=0.2):
        arena_width = int(np.sqrt(num_states))
        if N_inp % arena_width != 0:
            raise ValueError('Ensure N_inp is consistent with arena width')
        print(f'Place inputs generated from arena with width {arena_width}')
        self.N_inp = N_inp
        self.num_states = num_states
        self.arena_width = arena_width
        self.decay_constant = decay_constant
        N_inp_sqrt = int(np.sqrt(N_inp))

        inputs = np.zeros([num_states, N_inp])
        for s in range(num_states):
            peak = int(s / float(num_states) * N_inp)
            for n in range(N_inp):
                dist = distance2D(n, peak, N_inp_sqrt)
                inputs[s, n] = np.exp(-(dist/(N_inp*decay_constant)))
        inputs = inputs - np.mean(inputs, axis=1, keepdims=True)
        inputs = inputs / np.std(inputs, axis=1, keepdims=True)
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

from numpy.random import multivariate_normal
class PlaceInputsExp():
    def __init__(self, N_inp, num_states, decay_constant=0.4):
        self.N_inp = N_inp
        self.num_states = num_states
        self.decay_constant = decay_constant

        mean = np.zeros([num_states])
        cov = np.zeros([num_states, num_states])
        for s in range(num_states):
            for s2 in range(num_states):
                mindist = min(min(np.abs(s2-s), np.abs(s+num_states-s2)), np.abs(s-num_states-s2))
                cov[s, s2] = np.exp(-mindist/(decay_constant*num_states))

        inputs = multivariate_normal(mean, cov, size=[N_inp]).transpose()
        inputs = inputs - np.mean(inputs, axis=1, keepdims=True)
        inputs = inputs / np.std(inputs, axis=1, keepdims=True)

        self.inputs = inputs

    def get_inputs(self):
        return self.inputs
