import sys
sys.path.insert(1, '../core')
from Network import *

for i in range(3):
    if i == 0:
        transceiver = "fixed_rate"
    elif i == 1:
        transceiver = "flex_rate"
    else:
        transceiver = "shannon"

    net = Network("../resources/nodes_full_"+transceiver+".json")
    net.connect()
    net.create_weighted_paths()
    net.create_route_space()
    M = 1

    n_nodes = len(net.nodes)
    traffic_matrix = np.zeros((n_nodes, n_nodes))

    while np.count_nonzero(traffic_matrix) == 0:
        traffic_matrix = np.ones((n_nodes, n_nodes)) * 100e9 * M
        np.fill_diagonal(traffic_matrix, 0)
        conn_list = net.deploy_traffic_matrix(traffic_matrix)
        net.reset_network()
        M += 1

    print(transceiver.capitalize() + " DATA")
    print("Maximum M: " + str(M-2))
    print("Maximum bit rate: " + str(get_max_bitrate(conn_list)/1e9) + " Gpbs")
    print("Minimum bit rate: " + str(get_min_bitrate(conn_list)/1e9) + " Gbps")
    print("Average bit rate: " + str(get_average_bitrate(conn_list)/1e9) + " Gbps")
    plot_distribution(conn_list, "bit_rate", "../results/"+transceiver+"/bit_rate_distribution.png")
    plot_distribution(conn_list, "latency", "../results/"+transceiver+"/latency_distribution.png")

    net.route_space.to_csv("../results/"+transceiver+"/route_space.csv")
    net.weighted_paths.to_csv("../results/"+transceiver+"/weigthed_paths.csv")