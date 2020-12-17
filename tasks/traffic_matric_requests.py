import sys
sys.path.insert(1, '../core')
from Network import *
import random
import matplotlib.pyplot as plt

# networks generation
# NET 1 -> FIXED RATE TRANSCEIVER
net1 = Network("../resources/nodes_full_fixed_rate.json")
net1.connect()
net1.create_weighted_paths()
net1.create_route_space()

# NET 2 -> FLEX RATE TRANSCEIVER
net2 = Network("../resources/nodes_full_flex_rate.json")
net2.connect()
net2.create_weighted_paths()
net2.create_route_space()

# NET 3 -> SHANNON RATE TRANSCEIVER
net3 = Network("../resources/nodes_full_shannon.json")
net3.connect()
net3.create_weighted_paths()
net3.create_route_space()

n_nodes = len(net1.nodes)
traffic_matrix = np.ones((n_nodes, n_nodes))*150e9
np.fill_diagonal(traffic_matrix, 0)

connections = net1.deploy_traffic_matrix(traffic_matrix)
for conn in connections:
    print(conn.bit_rate)



print(traffic_matrix)
