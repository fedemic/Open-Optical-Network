import json
import pandas as pd
import random
import copy
from .Node import *
from .Line import *
from .Connection import *
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
                self.nodes[node_key].switching_matrix[connected_node] = copy.deepcopy(self.data_dict[node_key]['switching_matrix'][connected_node])

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

        def explore_node(current_node, destination_node_label, path_string):
            path_string += current_node.label
            for node_under_analysis in current_node.connected_nodes:
                if node_under_analysis == destination_node_label:
                    paths_list.append(path_string + destination_node_label)
                elif node_under_analysis not in path_string:
                    explore_node(self.nodes[node_under_analysis], destination_node_label, path_string)

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
        self.update_route_space()

    ###############################################################################
    # return path with highest snr between source and destination, considering a specific channel
    def find_best_snr(self, source, destination, channel):
        osnr_max = min(self.weighted_paths['OSNR'].values)
        best_path = ""

        for i, row in self.weighted_paths.iterrows():
            if row['path'][0] == source and row['path'][-1] == destination and row['OSNR'] > osnr_max:
                product = 1
                path = list(row['path'].split('->'))
                for j in range(len(path)-1):
                    line_label = path[j] + path[j+1]
                    product *= self.lines[line_label].state[channel]
                    if j != 0:
                        product *= self.nodes[path[j]].switching_matrix[path[j-1]][path[j+1]][channel]

                # Check availability by means of product
                if product == 1:
                    osnr_max = self.weighted_paths['OSNR'][i]
                    best_path = self.weighted_paths['path'][i]

        return best_path

    ###############################################################################
    # return path with lowest latency between source and destination, considering a specific channel
    def find_best_latency(self, source, destination, channel):
        latency_min = max(self.weighted_paths['latency'].values)
        best_path = ""

        for i, row in self.weighted_paths.iterrows():
            if row['path'][0] == source and row['path'][-1] == destination and row['latency'] < latency_min:
                product = 1
                path = list(row['path'].split('->'))
                for j in range(len(path) - 1):
                    line_label = path[j] + path[j + 1]
                    product *= self.lines[line_label].state[channel]
                    if j != 0:
                        product *= self.nodes[path[j]].switching_matrix[path[j-1]][path[j+1]][channel]

                # Check availability by means of product
                if product == 1:
                    latency_min = self.weighted_paths['latency'][i]
                    best_path = self.weighted_paths['path'][i]

        return best_path

    ###############################################################################
    # given a requested connection list, it deploys lightpaths with selected optimization
    def update_route_space(self):
        for i, row in self.route_space.iterrows():     # dataframe rows index
            partial_products = np.ones(N_CHANNELS).astype(int)      # initialization of partial products
            path = row['path'].split("->")

            for j in range(len(path)-1):    # nodes in path index
                line_label = path[j] + path[j+1]
                partial_products *= self.lines[line_label].state

                if j != 0:   # first node is not taken into account
                    partial_products *= self.nodes[path[j]].switching_matrix[path[j-1]][path[j+1]]

            for k in range(N_CHANNELS):     # k is the channel
                self.route_space.loc[i, 'CH_'+str(k)] = partial_products[k]

    ###############################################################################
    # calculate bit rate according to strategy and path
    def calculate_bit_rate(self, lightpath, strategy):
        path = "".join(lightpath.path)    # cast path from list to string
        formatted_path = ""            # adding the "->" for the formatted path
        for element in path:
            formatted_path += element + "->"
        formatted_path = formatted_path[:-2]

        rs = lightpath.rs
        if path == "":   # stream function could provide an empty path
            return 0

        gsnr_db = float(self.weighted_paths.loc[self.weighted_paths['path'] == formatted_path, 'OSNR'].values)
        gsnr = to_linear(gsnr_db)

        if strategy == 'fixed_rate':
            if gsnr >= 2*(special.erfcinv(2*BER_T)**2)*rs/BN:
                bit_rate = 100e9
            else:
                bit_rate = 0
        elif strategy == 'flex_rate':
            if gsnr < 4*BER_T*rs/BN:
                bit_rate = 0
            elif 2*(special.erfcinv(2*BER_T)**2)*rs/BN <= gsnr < 14/3*(special.erfcinv(3/2*BER_T)**2)*rs/BN:
                bit_rate = 100e9
            elif 14/3*(special.erfcinv(3/2*BER_T)**2)*rs/BN <= gsnr < 10*(special.erfcinv(8/3*BER_T)**2)*rs/BN:
                bit_rate = 200e9
            else:
                bit_rate = 400e9
        elif strategy == 'shannon':
            bit_rate = 2*RS*np.log2(1+gsnr*BN/rs)
        else:
            bit_rate = None

        return bit_rate

    ###############################################################################
    # It streams connections starting from a given traffic matrix
    def deploy_traffic_matrix(self, traffic_matrix):
        node_list = list(self.nodes.keys())
        initial_data = {}                     # dict for connection constructor
        signal_power = 1
        conn_list = []                        # list of deployed connections
        full_cells_list = []                  # list of all possible node-to-node requests
        conn_to_be_streamed = []              # list of connections to be streamed
        allocated_traffic = True
        matrix_fully_deployed = False

        for node1 in node_list:
            for node2 in node_list:
                full_cells_list.append([node1, node2])

        cells_list = list(full_cells_list)          # cast to list to have different pointer

        while matrix_fully_deployed is False and allocated_traffic is True:
            if cells_list == []:
                allocated_traffic = False
                cells_list = list(full_cells_list)
                self.stream(conn_to_be_streamed, signal_power, 'snr')
                for connection in conn_to_be_streamed:
                    if connection.snr is not None:
                        allocated_traffic = True
                        source_index = node_list.index(connection.input)
                        destination_index = node_list.index(connection.output)
                        if traffic_matrix[source_index, destination_index] >= connection.bit_rate:
                            traffic_matrix[source_index, destination_index] -= connection.bit_rate
                        else:
                            connection.bit_rate = traffic_matrix[source_index, destination_index]
                            traffic_matrix[source_index, destination_index] = 0

                conn_list.extend(conn_to_be_streamed)
                conn_to_be_streamed = []
                if np.count_nonzero(traffic_matrix) == 0:
                    matrix_fully_deployed = True

            inout_nodes = random.sample(cells_list, 1)
            cells_list.remove(inout_nodes[0])
            source_index = node_list.index(inout_nodes[0][0])
            destination_index = node_list.index(inout_nodes[0][1])

            if traffic_matrix[source_index, destination_index] != 0:
                initial_data["input"] = inout_nodes[0][0]
                initial_data["output"] = inout_nodes[0][1]
                initial_data["signal_power"] = signal_power

                connection = Connection(initial_data)
                conn_to_be_streamed.append(connection)

        return conn_list

    ###############################################################################
    # restore switching matrices state (adjacent channels occupation)
    def restore_switching_matrices(self):
        for node_key in self.nodes:
            for source_node in self.nodes[node_key].connected_nodes:
                for destination_node in self.nodes[node_key].connected_nodes:
                    if source_node != destination_node:
                        source_line_label = source_node + node_key
                        destination_line_label = node_key + destination_node

                        state_array = np.bitwise_or(self.lines[source_line_label].state, self.lines[destination_line_label].state)
                        mask = np.bitwise_and(state_array, self.data_dict[node_key]['switching_matrix'][source_node][destination_node])
                        self.nodes[node_key].switching_matrix[source_node][destination_node] = mask

    ###############################################################################
    # given a requested connection list, it deploys lightpaths with selected optimization
    def stream(self, connection_list, signal_power, optimize="latency"):
        for connection in connection_list:
            path = ""
            channel = -1
            bit_rate = 0
            lightpath = None   # lightpath initialization

            while (path == "" or bit_rate == 0) and channel <= N_CHANNELS-2:
                channel += 1                                                                            # next channel
                if optimize == "latency":
                    path = self.find_best_latency(connection.input, connection.output, channel)       # find a possible path
                elif optimize == "snr":
                    path = self.find_best_snr(connection.input, connection.output, channel)
                path = path.split("->")                                                                 # remove -> from the path
                lightpath = Lightpath(signal_power, list(path), channel)                                # create a lightpath with the found path
                bit_rate = self.calculate_bit_rate(lightpath, self.nodes[connection.input].transceiver)     # check the bitrate for the created lightpath

            if path == "" or bit_rate == 0:  # connection rejected
                connection.snr = None
                connection.latency = 0
            else:
                final_signal = self.propagate(lightpath)

                connection.signal_power = final_signal.signal_power
                connection.latency = final_signal.latency
                connection.snr = to_db(1/final_signal.inv_gsnr)
                connection.bit_rate = bit_rate

        self.restore_switching_matrices()
        self.update_route_space()

    ###############################################################################
    # Reset line state arrays and nodes switching matrices
    def reset_network(self):
        for node_key in self.nodes:
            for connected_node in self.nodes[node_key].connected_nodes:
                self.nodes[node_key].switching_matrix[connected_node] = copy.deepcopy(self.data_dict[node_key]['switching_matrix'][connected_node])
        for line_key in self.lines:
            self.lines[line_key].state = np.ones(N_CHANNELS).astype(int)

        self.create_route_space()
