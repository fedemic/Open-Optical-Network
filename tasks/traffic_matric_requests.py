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
M1 = 1

# NET 2 -> FLEX RATE TRANSCEIVER
net2 = Network("../resources/nodes_full_flex_rate.json")
net2.connect()
net2.create_weighted_paths()
net2.create_route_space()
M2 = 1

# NET 3 -> SHANNON RATE TRANSCEIVER
net3 = Network("../resources/nodes_full_shannon.json")
net3.connect()
net3.create_weighted_paths()
net3.create_route_space()
M3 = 1

n_nodes = len(net1.nodes)
fully_deployed = True

while fully_deployed == True:
    traffic_matrix = np.ones((n_nodes, n_nodes)) * 100e9 * M3
    np.fill_diagonal(traffic_matrix, 0)
    fully_deployed = net3.deploy_traffic_matrix(traffic_matrix)
    net3.reset_network()
    M3 += 1
print("Fixed-rate: " +str(M3-2))
