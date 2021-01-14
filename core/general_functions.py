from .constants import *
import matplotlib.pyplot as plt
import sys


def to_db(lin_value):
    return 10*np.log10(lin_value)


def to_linear(db_value):
    return 10**(db_value/10)


def to_alpha_linear(alpha_db):
    return alpha_db/(20*np.log10(e))


def get_max_bitrate(connections):
    max_br = 0
    for connection in connections:
        if connection.bit_rate > max_br:
            max_br = connection.bit_rate
    return max_br


def get_min_bitrate(connections):
    min_br = sys.maxsize
    for connection in connections:
        if connection.bit_rate < min_br and connection.bit_rate != 0:
            min_br = connection.bit_rate
    return min_br


def get_average_bitrate(connections):
    sum_br = 0
    num_of_zeros_bitrate = 0
    for connection in connections:
        if connection.bit_rate != 0:
            sum_br += connection.bit_rate
        else:
            num_of_zeros_bitrate += 1
    return sum_br/(len(connections)-num_of_zeros_bitrate)


def get_total_capacity(connections):
    sum_br = 0
    for connection in connections:
        sum_br += connection.bit_rate
    return sum_br


def plot_distribution(connections, parameter, filename):
    values = []
    for connection in connections:
        values.append(getattr(connection, parameter))

    plt.figure()
    plt.hist(values)
    plt.xlabel(parameter)
    plt.ylabel("Occurrences")
    plt.title(parameter + " distribution")
    plt.savefig(filename)
