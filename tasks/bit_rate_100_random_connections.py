import sys
sys.path.insert(1, '../core')
from Network import *
import random
import matplotlib.pyplot as plt

# network generation
net = Network("../resources/nodes_full_flex_rate.json")
net.connect()
net.create_weighted_paths()
net.create_route_space()

# 100 random requests creation
node_list = list(net.nodes.keys())
conn_list = []
initial_data = {}
signal_power = 1

for i in range(5):
    inout_nodes = random.sample(node_list, 2)

    initial_data["input"] = inout_nodes[0]
    initial_data["output"] = inout_nodes[1]
    initial_data["signal_power"] = signal_power

    conn_list.append(Connection(initial_data))

# request deployment optimizing SNR
net.stream(conn_list, signal_power, optimize="snr")

bit_rate_values = []
rejected_connections = 0
for connection in conn_list:
    if connection.snr != None:
        bit_rate_values.append(connection.bit_rate)
    else:
        rejected_connections += 1

print(bit_rate_values)
# results distribution plot
plt.figure()
plt.hist(bit_rate_values, color="r")
plt.xlabel("Bit-rate [Gbps]")
plt.ylabel("Occurences")
plt.title("Bit-rate Distribution")
plt.savefig('../results/bit_rate_distribution_fixed_rate.png')

bit_rate_average = np.mean(bit_rate_values)
total_capacity = np.sum(bit_rate_values)
print('Fixed rate transceiver results')
print('Number of rejected connections: ' + str(rejected_connections))
print('Average bit rate: ' + str(bit_rate_average))
print('Total capacity: ' + str(total_capacity))

