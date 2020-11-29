from Lightpath import *
from constants import *

class Node:
    def __init__(self, initial_data):
        self._label = initial_data['label']
        self._position = initial_data['position']
        self._connected_nodes = initial_data['connected_nodes']
        self._successive = {}   # dict of Line objects
        self._switching_matrix = None

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @label.setter
    def label(self, value):
        self._label = value

    @position.setter
    def position(self, value):
        self._position = value

    @connected_nodes.setter
    def connected_nodes(self, value):
        self._connected_nodes = value

    @successive.setter
    def successive(self, value):
        self._successive = value

    @switching_matrix.setter
    def switching_matrix(self, value):
        self._switching_matrix = value

    def propagate(self, propagated_object, previous_node):
        propagated_object.update_path()
        if len(propagated_object.path) != 0:
            if isinstance(propagated_object, Lightpath) and previous_node != None:    # Switching matrix update
                self.switching_matrix[previous_node][propagated_object.path[0]][propagated_object.channel] = 0
                if propagated_object.channel != 0:  # Channel 0 does not have a left adjacent channel
                    self.switching_matrix[previous_node][propagated_object.path[0]][propagated_object.channel-1] = 0
                if propagated_object.channel != N_CHANNELS-1: # Last channel does not have a right adjacent channel
                    self.switching_matrix[previous_node][propagated_object.path[0]][propagated_object.channel+1] = 0

            next_line_label = self.label + propagated_object.path[0]
            self.successive[next_line_label].propagate(propagated_object)
