import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Node import *
from Line import *
from Connection import *

class Network:
    def __init__(self, json_filepath):
        self._nodes = {}    # dict of Node objects
        self._lines = {}    # dict of Line objects
        self._weighted_paths = None    # Dataframe: |path|latency|SNR|noise|
        self._route_space = None       # Dataframe: |path|CH_0|...|CH_n|
        self._data_dict = None  # To store all the network data in the json file

        with open(json_filepath, "r") as read_file:
            self.data_dict = json.load(read_file)

        # _nodes and _lines initialization from json file
        node_dict = {}
        line_dict = {}

        for node_key in self.data_dict:
            node_dict['label'] = node_key
            node_dict['position'] = self.data_dict[node_key]['position']
            node_dict['connected_nodes'] = self.data_dict[node_key]['connected_nodes']

            self._nodes[node_key] = Node(node_dict)

            for conn_node in self.data_dict[node_key]['connected_nodes']:
                line_label = node_key + conn_node

                # line length evaluation
                position1 = np.array(self.data_dict[node_key]['position'])
                position2 = np.array(self.data_dict[conn_node]['position'])
                line_length = np.linalg.norm(position1 - position2)

                line_dict["label"] = line_label
                line_dict["length"] = line_length

                self._lines[line_label] = Line(line_dict)

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

    @property
    def data_dict(self):
        return self._data_dict

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @lines.setter
    def lines(self, value):
        self._lines = value

    @weighted_paths.setter
    def weighted_paths(self, value):
        self._weighted_paths = value

    @route_space.setter
    def route_space(self, value):
        self._route_space = value

    @data_dict.setter
    def data_dict(self, value):
        self._data_dict = value

    ###############################################################################
    # update of successive dictionaries of nodes and lines and initialize switching matrix
    def connect(self):
        for node_key in self.nodes:
            self.nodes[node_key].switching_matrix = {}  # initialization to empty dict for each node

            # successive elements definition
            for connected_node in self.nodes[node_key].connected_nodes:
                line_label = node_key + connected_node
                self.nodes[node_key].successive[line_label] = self.lines[line_label]
                self.lines[line_label].successive[connected_node] = self.nodes[connected_node]

                self.nodes[node_key].switching_matrix[connected_node] = self.data_dict[node_key]['switching_matrix'][connected_node]

    ###############################################################################
    # network map plot
    def draw(self):
        plt.figure()
        plt.grid()
        plt.xlabel("Position x [km]")
        plt.ylabel("Position y [km]")
        plt.title("Network map")

        for line_key in self.lines:
            # source node
            x1 = self.nodes[line_key[0]].position[0]
            y1 = self.nodes[line_key[0]].position[1]
            # destination node
            x2 = self.nodes[line_key[1]].position[0]
            y2 = self.nodes[line_key[1]].position[1]

            plt.plot([x1, x2], [y1, y2], linewidth=2, color="k")

        for node_key in self.nodes:
            x_value = self.nodes[node_key].position[0]
            y_value = self.nodes[node_key].position[1]
            plt.plot(x_value, y_value, "o", label=node_key, markersize=12)

        plt.legend()
        plt.savefig('../results/network_map.png')

    ###############################################################################
    # find all possible paths between source and destination
    def find_path(self, source, destination):
        path = ""
        paths_list = []

        def explore_node(current_node, destination, path):
            path += current_node.label
            for node_under_analysis in current_node.connected_nodes:
                if node_under_analysis == destination:
                    paths_list.append(path + destination)
                elif node_under_analysis not in path:
                    explore_node(self.nodes[node_under_analysis], destination, path)

        if source != destination:
            explore_node(self.nodes[source], destination, path)
            return paths_list
        else:
            return source

    ###############################################################################
    # calls the propagate method of first node in the propagated_object's path
    def propagate(self, propagated_object):
        start_node_key = propagated_object.path[0]
        self.nodes[start_node_key].propagate(propagated_object, None)

        return propagated_object

    ###############################################################################
    # generate weighted_paths dataframe considering all possible paths
    def create_paths_database(self):
        db_dict = {"path": [],
                   "latency": [],
                   "noise": [],
                   "OSNR": []}

        for start_node in self.nodes:
            for destination_node in self.nodes:
                if start_node != destination_node:
                    path_list = self.find_path(start_node, destination_node)
                    for path in path_list:
                        signal = SignalInformation(1e-3, list(path))    # SignalInformation object is used just to get
                        final_signal = self.propagate(signal)           # the effects of propagation without
                        formatted_path = ""                             # considering channel occupation

                        for element in path:
                            formatted_path += element + "->"
                        formatted_path = formatted_path[:-2]

                        db_dict["path"].append(formatted_path)
                        db_dict["latency"].append(final_signal.latency)
                        db_dict["noise"].append(final_signal.noise_power)
                        osnr = 10 * np.log10(final_signal.signal_power / final_signal.noise_power)
                        db_dict["OSNR"].append(osnr)

        self.weighted_paths = pd.DataFrame(db_dict)

    ###############################################################################
    # generate route_space dataframe considering all possible paths
    def create_route_space(self):
        db_dict = {"path": []}
        for channel in range(N_CHANNELS):
            db_dict["CH_"+str(channel)] = []

        for start_node in self.nodes:
            for destination_node in self.nodes:
                if start_node != destination_node:
                    path_list = self.find_path(start_node, destination_node)
                    for path in path_list:

                        formatted_path = ""
                        for element in path:
                            formatted_path += element + "->"
                        formatted_path = formatted_path[:-2]

                        db_dict['path'].append(formatted_path)
                        for channel in range(N_CHANNELS):
                            db_dict['CH_'+str(channel)].append(0)

        self.route_space = pd.DataFrame(db_dict)

    ###############################################################################
    # return path with highest snr between source and destination, considering a specific channel
    def find_best_snr(self, source, destination, channel):
        indexes_range = self.weighted_paths.index
        osnr_max = min(self.weighted_paths['OSNR'].values)
        best_path = ""

        for i in indexes_range:
            if self.weighted_paths['path'][i][0] == source and self.weighted_paths['path'][i][-1] == destination and self.weighted_paths['OSNR'][i] > osnr_max:
                lightpath_available = True  # initialization
                nodes_label_list = self.weighted_paths['path'][i].split("->")

                for j in range(len(nodes_label_list) - 1):
                    line_label = nodes_label_list[j] + nodes_label_list[j + 1]
                    if self.lines[line_label].state[channel] == 0:
                        lightpath_available = False

                if lightpath_available == True:
                    osnr_max = self.weighted_paths['OSNR'][i]
                    best_path = self.weighted_paths['path'][i]

        return best_path

    ###############################################################################
    # return path with lowest latency between source and destination, considering a specific channel
    def find_best_latency(self, source, destination, channel):
        indexes_range = self.weighted_paths.index
        latency_min = max(self.weighted_paths['latency'].values)
        best_path = ""

        for i in indexes_range:
            if self.weighted_paths['path'][i][0] == source and self.weighted_paths['path'][i][-1] == destination and self.weighted_paths['latency'][i] < latency_min:
                lightpath_available = True  # initialization
                nodes_label_list = self.weighted_paths['path'][i].split("->")

                for j in range(len(nodes_label_list)-1):
                    line_label = nodes_label_list[j] + nodes_label_list[j+1]
                    if self.lines[line_label].state[channel] == 0:
                        lightpath_available = False

                if lightpath_available == True:
                    latency_min = self.weighted_paths['latency'][i]
                    best_path = self.weighted_paths['path'][i]
        return best_path

    ###############################################################################
    # given a requested connection list, it deploys lightpaths with selected optimization
    def update_route_space(self):
        indexes_range = self.route_space.index
        for i in indexes_range:     # dataframe rows index
            partial_products = np.ones(N_CHANNELS).astype(int)      # initialization of partial products
            nodes_label_list = self.route_space['path'][i].split("->")

            for j in range(len(nodes_label_list)-1):    # nodes in path index
                line_label = nodes_label_list[j] + nodes_label_list[j + 1]
                partial_products *= self.lines[line_label].state

                if j != 0: # first node is not taken into account
                    partial_products *= self.nodes[nodes_label_list[j]].switching_matrix[nodes_label_list[j-1]][nodes_label_list[j+1]]

            for k in range(N_CHANNELS):     # k is the channel
                self.route_space.loc[i, 'CH_'+str(k)] = partial_products[k]

    ###############################################################################
    # given a requested connection list, it deploys lightpaths with selected optimization
    def stream(self, connection_list, signal_power, optimize="latency"):
        for connection in connection_list:
            path = ""
            channel = -1
            if optimize == "latency":
                while path == "" and channel <= N_CHANNELS-2:
                    channel += 1
                    path = self.find_best_latency(connection.input, connection.output, channel)
            elif optimize == "snr":
                while path == "" and channel <= N_CHANNELS-2:
                    channel += 1
                    path = self.find_best_snr(connection.input, connection.output, channel)

            if path == "":
                connection.snr = None
                connection.latency = 0
            else:
                path = path.split("->")
                lightpath = Lightpath(signal_power, list(path), channel)    # Lightpath object is used to consider channel info
                final_signal = self.propagate(lightpath)

                connection.signal_power = final_signal.signal_power
                connection.latency = final_signal.latency
                connection.snr = 10*np.log10(final_signal.signal_power/final_signal.noise_power)

                self.update_route_space()