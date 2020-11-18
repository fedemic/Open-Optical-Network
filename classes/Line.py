from Lightpath import *
from scipy.constants import c

class Line:
    def __init__(self, initial_data):
        self._label = initial_data["label"]
        self._length = initial_data["length"]
        self._successive = {}   # dict of Node objects
        self._state = []        # channel availabily list

        for i in range(10):
            self._state.append("free")

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

    def latency_generation(self):
        return self.length/(c*2/3)

    def noise_generation(self, signal):
        return 1e-3 * signal.signal_power * self.length

    def propagate(self, propagated_object):
        propagated_object.update_latency(self.latency_generation())
        propagated_object.update_noise_power(self.noise_generation(propagated_object))

        next_node_label = propagated_object.path[0]
        if isinstance(propagated_object, Lightpath):    # if Lightpath object then manage channels
            self.state[propagated_object.channel] = "occupied"
        self.successive[next_node_label].propagate(propagated_object)
