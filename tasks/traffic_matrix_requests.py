import sys
sys.path.insert(1, '../')
from core.Network import *

for i in range(3):
    if i == 0:
        transceiver = "fixed_rate"
        #M = 4
    elif i == 1:
        transceiver = "flex_rate"
        #M = 16
    else:
        transceiver = "shannon"
        #M = 17

    net = Network("../resources/nodes_full_"+transceiver+".json")
    net.connect()
    net.create_weighted_paths()
    net.create_route_space()
    M = 1

    conn_list = []
    n_nodes = len(net.nodes)
    traffic_matrix = np.zeros((n_nodes, n_nodes))

    while np.count_nonzero(traffic_matrix) == 0:
        traffic_matrix = np.ones((n_nodes, n_nodes)) * 100e9 * M
        np.fill_diagonal(traffic_matrix, 0)
        conn_list = net.deploy_traffic_matrix(traffic_matrix)
        net.reset_network()
        M += 1

    with open("../results/traffic_matrix/"+transceiver+"/results.txt", 'w') as file:
        file.write(transceiver.capitalize() + " DATA\n")
        file.write("Maximum M: " + str(M-2)+"\n")
        file.write("Maximum bit rate: " + str(get_max_bitrate(conn_list)/1e9) + " Gpbs\n")
        file.write("Minimum bit rate: " + str(get_min_bitrate(conn_list)/1e9) + " Gbps\n")
        file.write("Average bit rate: " + str(get_average_bitrate(conn_list)/1e9) + " Gbps\n")
        file.write("Total capacity: " + str(get_total_capacity(conn_list) / 1e9) + " Gbps")

    print(transceiver.capitalize() + " DATA")
    print("Maximum M: " + str(M-2))
    print("Maximum bit rate: " + str(get_max_bitrate(conn_list)/1e9) + " Gpbs")
    print("Minimum bit rate: " + str(get_min_bitrate(conn_list)/1e9) + " Gbps")
    print("Average bit rate: " + str(get_average_bitrate(conn_list)/1e9) + " Gbps")
    print("Total capacity: " + str(get_total_capacity(conn_list) / 1e9) + " Gbps")

    plot_distribution(conn_list, "bit_rate", "../results/traffic_matrix/"+transceiver+"/bit_rate_distribution.png")
    plot_distribution(conn_list, "latency", "../results/traffic_matrix/"+transceiver+"/latency_distribution.png")

    net.route_space.to_csv("../results/traffic_matrix/"+transceiver+"/route_space.csv")
    net.weighted_paths.to_csv("../results/traffic_matrix/"+transceiver+"/weighted_paths.csv")
