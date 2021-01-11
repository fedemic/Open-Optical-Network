import numpy as np
from constants import *
import matplotlib.pyplot as plt

def to_db(lin_value):
    return 10*np.log10(lin_value)

def to_linear(db_value):
    return 10**(db_value/10)

def to_alpha_linear(alpha_db):
    return alpha_db/(20*np.log10(e))

def get_max_bitrate(connections):
    max = 0
    for connection in connections:
        if connection.bit_rate > max:
            max = connection.bit_rate
    return max

def get_min_bitrate(connections):
    min = 1e20
    for connection in connections:
        if connection.bit_rate < min and connection.bit_rate != 0:
            min = connection.bit_rate
    return min

def get_average_bitrate(connections):
    sum = 0
    num_of_zeros_bitrate = 0
    for connection in connections:
        if connection.bit_rate != 0:
            sum += connection.bit_rate
        else:
            num_of_zeros_bitrate += 1
    return sum/(len(connections)-num_of_zeros_bitrate)

def get_total_capacity(connections):
    sum = 0
    for connection in connections:
            sum += connection.bit_rate
    return sum

def plot_distribution(connections, parameter, filename):
    values = []
    for connection in connections:
        values.append(getattr(connection, parameter))

    plt.figure()
    plt.hist(values)
    plt.xlabel(parameter)
    plt.ylabel("Occurences")
    plt.title(parameter + " distribution")
    plt.savefig(filename)