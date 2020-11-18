import sys
sys.path.insert(1, '../classes')
from Network import *
import random
import matplotlib.pyplot as plt

# network generation
net = Network("../resources/nodes.json")
net.connect()
net.draw()
net.create_paths_database()
net.create_route_space()

# 100 random requests creation
node_list = list(net.nodes.keys())
conn_list = []
initial_data = {}
signal_power = 1

for i in range(100):
    inout_nodes = random.sample(node_list, 2)

    initial_data["input"] = inout_nodes[0]
    initial_data["output"] = inout_nodes[1]
    initial_data["signal_power"] = signal_power

    conn_list.append(Connection(initial_data))

# request deployment optimizing latency
net.stream(conn_list, signal_power, optimize="latency")

latency_values = []
for connection in conn_list:
    latency_values.append(connection.latency)

# request deployment optimizing SNR
net.stream(conn_list, signal_power, optimize="snr")

snr_values = []
for connection in conn_list:
    if connection.snr != None:
        snr_values.append(connection.snr)

# results distribution plot
plt.figure()
plt.hist(latency_values)
plt.xlabel("Latency [s]")
plt.ylabel("Occurences")
plt.title("Latency Distribution")
plt.savefig('../results/latency_distribution.png')

plt.figure()
plt.hist(snr_values, color="r")
plt.xlabel("SNR [dB]")
plt.ylabel("Occurences")
plt.title("SNR Distribution")
plt.savefig('../results/snr_distribution.png')


# NON RICHIESTA
net.route_space.to_csv('../results/route_space.csv')
net.weighted_paths.to_csv('../results/weigthed_paths.csv')
