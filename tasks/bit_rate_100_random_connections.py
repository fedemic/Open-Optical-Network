import sys
sys.path.insert(1, '../')
from core.Network import *

for i in range(3):
    if i == 0:
        transceiver = "fixed_rate"
    elif i == 1:
        transceiver = "flex_rate"
    else:
        transceiver = "shannon"

    net = Network("../resources/nodes_full_" + transceiver + ".json")
    net.connect()
    net.create_weighted_paths()
    net.create_route_space()

    initial_data = {}
    signal_power = 1e-3

    # random connections generation on the first iteration (so the same connections are employed in the analysis)
    if i == 0:
        conn_list = []
        # 100 random requests creation
        node_list = list(net.nodes.keys())

        for j in range(N_CONNECTIONS):
            inout_nodes = random.sample(node_list, 2)

            initial_data["input"] = inout_nodes[0]
            initial_data["output"] = inout_nodes[1]
            initial_data["signal_power"] = signal_power

            conn_list.append(Connection(initial_data))

    # request deployment optimizing SNR for each net
    deployed_conn_list = list(conn_list)
    net.stream(deployed_conn_list, signal_power, optimize="snr")

    bit_rate_values = []
    rejected_connections = 0

    for connection in deployed_conn_list:
        if connection.snr is not None:
            bit_rate_values.append(connection.bit_rate)
        else:
            rejected_connections += 1

    # results distribution plot
    plt.figure()
    plt.hist(bit_rate_values, color="r")
    plt.xlabel("Bit-rate [Gbps]")
    plt.ylabel("Occurrences")
    plt.title(transceiver+" transceiver")
    plt.savefig("../results/bitrate_100_random/"+transceiver+"_bit_rate_distributions.png")

    bit_rate_average = get_average_bitrate(deployed_conn_list)
    total_capacity = get_total_capacity(deployed_conn_list)

    # results file
    with open("../results/bitrate_100_random/results_"+transceiver+".txt", 'w') as file:
        file.write(transceiver + ' transceiver results\n')
        file.write('Number of rejected connections: ' + str(rejected_connections) + "\n")
        file.write('Average bit rate: ' + str(bit_rate_average) + "\n")
        file.write('Total capacity: ' + str(total_capacity) + "\n")

    print(transceiver + ' transceiver results')
    print('Number of rejected connections: ' + str(rejected_connections))
    print('Average bit rate: ' + str(bit_rate_average))
    print('Total capacity: ' + str(total_capacity) + "\n")
