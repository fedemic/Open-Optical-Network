import numpy as np
from Lightpath import *
from constants import *

class Line:
    def __init__(self, initial_data):
        self._label = initial_data["label"]
        self._length = initial_data["length"]
        self._successive = {}   # dict of Node objects
        self._state = np.ones(N_CHANNELS).astype(int)        # channel availability numpy array
        self._n_amplifiers = np.ceil(self._length*1e-3/80)-1    # ---80km---AMP---80km---
        self._gain = initial_data['amp_gain']
        self._noise_figure = initial_data['amp_noise_figure']
        self._alpha = initial_data['alpha']         # [dB/km]
        self._beta_2 = initial_data['beta_2']       # [ps^2/km]
        self._gamma = initial_data['gamma']         # [(W*m)^-1]

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def state(self):
        return self._state

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta_2(self):
        return self._beta_2

    @property
    def gamma(self):
        return self._gamma

    @label.setter
    def label(self, value):
        self._label = value

    @length.setter
    def length(self, value):
        self._length = value

    @successive.setter
    def successive(self, value):
        self._successive = value

    @state.setter
    def state(self, value):
        self._state = value

    @n_amplifiers.setter
    def n_amplifiers(self, value):
        self._n_amplifiers = value

    @gain.setter
    def gain(self, value):
        self._gain = value

    @noise_figure.setter
    def noise_figure(self, value):
        self._noise_figure = value

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @beta_2.setter
    def beta_2(self, value):
        self._beta_2 = value

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def latency_generation(self):
        return self.length/(c*2/3)

    def noise_generation(self, signal):
        return 1e-9 * signal.signal_power * self.length

    def propagate(self, propagated_object):
        propagated_object.update_latency(self.latency_generation())
        propagated_object.update_noise_power(self.noise_generation(propagated_object))

        next_node_label = propagated_object.path[0]
        if isinstance(propagated_object, Lightpath):    # if Lightpath object then manage channels
            self.state[propagated_object.channel] = 0
        self.successive[next_node_label].propagate(propagated_object, self.label[0])

    def ase_generation(self):
        nf_lin = to_linear(self.noise_figure)
        gain_lin = to_linear(self.gain)

        return self.n_amplifiers*h*F*BN*nf_lin*(gain_lin-1)


