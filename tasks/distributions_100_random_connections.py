import sys
sys.path.insert(1, '../core')
from Network import *

for i in range(2):
    if i == 0:
        file_name = "nodes_not_full"
    else:
        file_name = "nodes_full"

    # network generation
    net = Network("../resources/"+file_name+".json")
    net.connect()
    net.draw()
    net.create_weighted_paths()
    net.create_route_space()

    # random connections generation on the first iteration (so the same connections are employed in the analysis)
    if i == 0:
        # 100 random requests creation
        node_list = list(net.nodes.keys())
        conn_list = []
        initial_data = {}
        signal_power = 1

        for j in range(N_CONNECTIONS):
            inout_nodes = random.sample(node_list, 2)

            initial_data["input"] = inout_nodes[0]
            initial_data["output"] = inout_nodes[1]
            initial_data["signal_power"] = signal_power

            conn_list.append(Connection(initial_data))

    # request deployment optimizing latency
    deployed_conn_list = list(conn_list)
    net.stream(deployed_conn_list, signal_power, optimize="latency")

    latency_values = []
    for connection in deployed_conn_list:
        latency_values.append(connection.latency)

    # reset of the lines states and switching matrices
    net.reset_network()

    # request deployment optimizing SNR
    deployed_conn_list = list(conn_list)
    net.stream(deployed_conn_list, signal_power, optimize="snr")

    snr_values = []
    for connection in deployed_conn_list:
        if connection.snr != None:
            snr_values.append(connection.snr)

    # results distribution plot
    plt.figure()
    plt.hist(latency_values)
    plt.xlabel("Latency [s]")
    plt.ylabel("Occurences")
    plt.title("Latency Distribution")
    plt.savefig("../results/snr_latency_100_random/"+file_name+"/latency_distribution.png")

    plt.figure()
    plt.hist(snr_values, color="r")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Occurences")
    plt.title("SNR Distribution")
    plt.savefig("../results/snr_latency_100_random/"+file_name+"/snr_distribution.png")

    # Not requested
    net.route_space.to_csv("../results/snr_latency_100_random/"+file_name+"/route_space.csv")
    net.weighted_paths.to_csv("../results/snr_latency_100_random/"+file_name+"/weighted_paths.csv")
