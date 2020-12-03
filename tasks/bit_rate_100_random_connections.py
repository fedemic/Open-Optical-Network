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

# 100 random requests creation
node_list = list(net1.nodes.keys())
conn_list1 = []
conn_list2 = []
conn_list3 = []
initial_data = {}
signal_power = 1

for i in range(N_CONNECTIONS):
    inout_nodes = random.sample(node_list, 2)

    initial_data["input"] = inout_nodes[0]
    initial_data["output"] = inout_nodes[1]
    initial_data["signal_power"] = signal_power

    conn_list1.append(Connection(initial_data))
    conn_list2.append(Connection(initial_data))
    conn_list3.append(Connection(initial_data))

# request deployment optimizing SNR for each net
net1.stream(conn_list1, signal_power, optimize="snr")
net2.stream(conn_list2, signal_power, optimize="snr")
net3.stream(conn_list3, signal_power, optimize="snr")

bit_rate_values1 = []
bit_rate_values2 = []
bit_rate_values3 = []
rejected_connections1 = 0
rejected_connections2 = 0
rejected_connections3 = 0

for connection in conn_list1:
    if connection.snr != None:
        bit_rate_values1.append(connection.bit_rate)
    else:
        rejected_connections1 += 1

for connection in conn_list2:
    if connection.snr != None:
        bit_rate_values2.append(connection.bit_rate)
    else:
        rejected_connections2 += 1

for connection in conn_list3:
    if connection.snr != None:
        bit_rate_values3.append(connection.bit_rate)
    else:
        rejected_connections3 += 1

# results distribution plot
plt.figure()
plt.subplot(131)
plt.hist(bit_rate_values1, color="r")
plt.xlabel("Bit-rate [Gbps]")
plt.ylabel("Occurences")
plt.title("Fixed rate transceiver")
plt.subplot(132)
plt.hist(bit_rate_values2, color="r")
plt.xlabel("Bit-rate [Gbps]")
plt.ylabel("Occurences")
plt.title("Flex rate transceiver")
plt.subplot(133)
plt.hist(bit_rate_values3, color="r")
plt.xlabel("Bit-rate [Gbps]")
plt.ylabel("Occurences")
plt.title("Shannon transceiver")
plt.savefig('../results/bit_rate_distributions.png')

bit_rate_average1 = np.mean(bit_rate_values1)
total_capacity1 = np.sum(bit_rate_values1)
bit_rate_average2 = np.mean(bit_rate_values2)
total_capacity2 = np.sum(bit_rate_values2)
bit_rate_average3 = np.mean(bit_rate_values3)
total_capacity3 = np.sum(bit_rate_values3)
print('Fixed rate transceiver results')
print('Number of rejected connections: ' + str(rejected_connections1))
print('Average bit rate: ' + str(bit_rate_average1))
print('Total capacity: ' + str(total_capacity1) +"\n")

print('Flex rate transceiver results')
print('Number of rejected connections: ' + str(rejected_connections2))
print('Average bit rate: ' + str(bit_rate_average2))
print('Total capacity: ' + str(total_capacity2) +"\n")

print('Shannon transceiver results')
print('Number of rejected connections: ' + str(rejected_connections3))
print('Average bit rate: ' + str(bit_rate_average3))
print('Total capacity: ' + str(total_capacity3) +"\n")
