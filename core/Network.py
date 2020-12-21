import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from Node import *
from Line import *
from Connection import *
from constants import *
from general_functions import *
from scipy import special


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

            if 'transceiver' in self.data_dict[node_key].keys():   # control presence of transceiver field
                node_dict['transceiver'] = self.data_dict[node_key]['transceiver']
            else:
                node_dict['transceiver'] = 'fixed_rate'

            self._nodes[node_key] = Node(node_dict)

            for conn_node in self.data_dict[node_key]['connected_nodes']:
                line_label = node_key + conn_node

                # line length evaluation
                position1 = np.array(self.data_dict[node_key]['position'])
                position2 = np.array(self.data_dict[conn_node]['position'])
                line_length = np.linalg.norm(position1 - position2)

                line_dict["label"] = line_label
                line_dict["length"] = line_length

                line_dict['amp_gain'] = AMP_GAIN
                line_dict['amp_noise_figure'] = AMP_NF

                line_dict['alpha'] = ALPHA
                line_dict['beta_2'] = BETA_2
                line_dict['gamma'] = GAMMA

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

                # Switching matrices initialization
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
            x1 = self.nodes[line_key[0]].position[0]*1e-3
            y1 = self.nodes[line_key[0]].position[1]*1e-3
            # destination node
            x2 = self.nodes[line_key[1]].position[0]*1e-3
            y2 = self.nodes[line_key[1]].position[1]*1e-3

            plt.plot([x1, x2], [y1, y2], linewidth=2, color="k")

        for node_key in self.nodes:
            x_value = self.nodes[node_key].position[0]*1e-3
            y_value = self.nodes[node_key].position[1]*1e-3
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
    def create_weighted_paths(self):
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
                        osnr = to_db(1/final_signal.inv_gsnr)
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
                            db_dict['CH_'+str(channel)].append(1)

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

                # Check availability in route_space dataframe
                if int(self.route_space.loc[self.route_space['path'] == self.weighted_paths['path'][i], 'CH_'+str(channel)].values) == 0:
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

                # Check availability in route_space dataframe
                if int(self.route_space.loc[self.route_space['path'] == self.weighted_paths['path'][i], 'CH_' + str(channel)].values) == 0:
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
    # calculate bit rate according to strategy and path
    def calculate_bit_rate(self, lightpath, strategy):

        path = "".join(lightpath.path) # cast path from list to string
        formatted_path = ""            # adding the "->" for the formatted path
        for element in path:
            formatted_path += element + "->"
        formatted_path = formatted_path[:-2]

        rs = lightpath.rs
        if path == "":   # stream function could provide an empty path
            return 0

        GSNR_dB = float(self.weighted_paths.loc[self.weighted_paths['path'] == formatted_path, 'OSNR'].values)
        GSNR = to_linear(GSNR_dB)
        bit_rate = 0

        if strategy == 'fixed_rate':
            if GSNR >= 2*(special.erfcinv(2*BER_T)**2)*rs/BN:
                bit_rate = 100e9
            else:
                bit_rate = 0
        elif strategy == 'flex_rate':
            if GSNR < 4*BER_T*rs/BN:
                bit_rate = 0
            elif GSNR >= 2*(special.erfcinv(2*BER_T)**2)*rs/BN and GSNR < 14/3*(special.erfcinv(3/2*BER_T)**2)*rs/BN:
                bit_rate = 100e9
            elif GSNR >= 14/3*(special.erfcinv(3/2*BER_T)**2)*rs/BN and GSNR < 10*(special.erfcinv(8/3*BER_T)**2)*rs/BN:
                bit_rate = 200e9
            else:
                bit_rate = 400e9
        elif strategy == 'shannon':
            bit_rate = 2*RS*np.log2(1+GSNR*BN/rs)
        else:
            bit_rate = None

        return bit_rate

    ###############################################################################
    def deploy_traffic_matrix(self, traffic_matrix):
        node_list = list(self.nodes.keys())
        matrix_fully_deployed = False
        initial_data = {}
        signal_power = 1
        conn_list = []
        n_nodes = len(self.nodes)
        empty_traffic_matrix = np.zeros((n_nodes, n_nodes))

        while matrix_fully_deployed == False:
            inout_nodes = random.sample(node_list, 2)
            source_index = node_list.index(inout_nodes[0])
            destination_index = node_list.index(inout_nodes[1])

            requested_bit_rate = traffic_matrix[source_index, destination_index]
            if requested_bit_rate != 0:
                initial_data["input"] = inout_nodes[0]
                initial_data["output"] = inout_nodes[1]
                initial_data["signal_power"] = signal_power

                conn_list.append(Connection(initial_data))
                self.stream([conn_list[-1]], signal_power, 'snr')
                if conn_list[-1].snr != None:
                    if traffic_matrix[source_index, destination_index] >= conn_list[-1].bit_rate:
                        traffic_matrix[source_index, destination_index] -= conn_list[-1].bit_rate
                    else:
                        conn_list[-1].bit_rate = traffic_matrix[source_index, destination_index]
                        traffic_matrix[source_index, destination_index] = 0

            if np.count_nonzero(traffic_matrix) == 0:
                matrix_fully_deployed = True

        return conn_list



    ###############################################################################
    # given a requested connection list, it deploys lightpaths with selected optimization
    def stream(self, connection_list, signal_power, optimize="latency"):
        for connection in connection_list:
            path = ""
            channel = -1
            bit_rate = 0
            if optimize == "latency":
                while (path == "" or bit_rate == 0) and channel <= N_CHANNELS-2:
                    channel += 1                                                                            # next channel
                    path = self.find_best_latency(connection.input, connection.output, channel)             # find a possible path
                    path = path.split("->")                                                                 # remove -> from the path
                    lightpath = Lightpath(signal_power, list(path), channel)                                # create a lightpath with the found path
                    bit_rate = self.calculate_bit_rate(lightpath, self.nodes[connection.input].transceiver) # check the bitrate for the created lightpath
            elif optimize == "snr":
                while (path == "" or bit_rate == 0) and channel <= N_CHANNELS-2:
                    channel += 1
                    path = self.find_best_snr(connection.input, connection.output, channel)
                    path = path.split("->")
                    lightpath = Lightpath(signal_power, list(path), channel)
                    bit_rate = self.calculate_bit_rate(lightpath, self.nodes[connection.input].transceiver)


            if path == "" or bit_rate == 0:  # connection rejected
                connection.snr = None
                connection.latency = 0
            else:
                final_signal = self.propagate(lightpath)

                connection.signal_power = final_signal.signal_power
                connection.latency = final_signal.latency
                connection.snr = to_db(final_signal.signal_power/final_signal.noise_power)
                connection.bit_rate = bit_rate

                self.update_route_space()



        # Controllo e reset in deploy_traffic_matrix
        """""
        # Restore the original switching matrices
        for node_key in self.nodes:
            for connected_node in self.nodes[node_key].connected_nodes:
                self.nodes[node_key].switching_matrix[connected_node] = self.data_dict[node_key]['switching_matrix'][connected_node]

        # Restore the original line occupation arrays
        for line_key in self.lines:
            self.lines[line_key].state = np.ones(N_CHANNELS).astype(int)
        """""