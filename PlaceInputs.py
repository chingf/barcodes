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
        inputs = inputs - np.mean(inputs, axis=1, keepdims=True)
        inputs = inputs / np.std(inputs, axis=1, keepdims=True)
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

