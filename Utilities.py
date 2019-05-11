"""


"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Common import *


def sliding_windows(data, seq_length):
    """

    :param data:
    :param seq_length:
    :return:
    """
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def plot_trajectory(data):
    """

    :param x:
    :param y:
    :return:
    """
    if isinstance(data, tuple):
        x = data[0]
        y = data[1]
        plt.plot(x,y, label='Trajectory')

    else:
        plt.plot(data, label='Trajectory')

    plt.show()


def scale_data(data):
    """

    :param data:
    :return:
    """
    sc = MinMaxScaler()
    return sc, sc.fit_transform(data)


def get_all_data(filename):
    """

    :param filename:
    :return:
    """
    df = pd.read_csv(filename)
    #print(df.head())
    #print(df.shape)

    return df


def get_target_data(df, target_id):
    """

    :param df:
    :param target_id:
    :return:
    """
    df = df[df.id == target_id]

    return df


def run():
    """
    Start of this module
    :return:
    """
    print("Module of Utilities")
    df = get_all_data("data/prepared_data.csv")
    target_df = get_target_data(df, TARGET_ID)



if __name__ == "__main__":
    run()