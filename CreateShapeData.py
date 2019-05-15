""""

"""
import Utilities as utils

from sklearn.datasets import make_circles, make_moons
import numpy as np
import pandas as pd
import math
import  sys

DATA_POINTS = 10000


def define_shape(x, y, train=True):
    """

    :param x:
    :param y:
    :return:
    """
    if train:
        l = 0
    else:
        l = 1

    df_x = pd.DataFrame(data=x)
    # print(df_x)
    df_y = pd.DataFrame(data=y)
    # print(df_y)

    df = pd.concat([df_x, df_y], axis=1)
    df.columns = ['a', 'b', 'c']

    df['a'] = df['a'].apply(lambda x: x+100+l)
    df['b'] = df['b'].apply(lambda x: x + 200+l)

    df = df[df.c == 0]
    dataset = df.values

    return dataset


def create_shape(shape_type, train=True):
    dataset = []
    shape = ''

    if train:
        l = 0

    else:
        l = 1

    if shape_type == 1:
        shape = "line"
        print(shape.upper())

        for i in range(DATA_POINTS):
            x = 100 + (i/100) + l
            y = 200 + l
            coordinates = [x,y]
            dataset.append(coordinates)

    elif shape_type == 2:
        shape = "moon"
        print(shape.upper())
        x, y= make_moons(n_samples=DATA_POINTS, shuffle=False, noise=None, random_state=l)
        dataset = define_shape(x, y, train)

        dataset = dataset[:,[0,1]]

    elif shape_type == 3:
        shape = "L"
        print(shape.upper())
        print("For this shape: 70% is going down, and 30% is going right")
        down_points = int(DATA_POINTS * 0.7)
        right_points = DATA_POINTS - down_points
        #print(down_points)
        #print(right_points)

        for i in range(down_points):
            x = 100 + l
            y = 200 - (i/100) + l
            coordinates = [x,y]
            dataset.append(coordinates)

        #print(dataset)
        last_point = dataset[-1]
        #print(last_point)

        for i in range(right_points):
            x = last_point[0] + ((i+1)/100)
            y = last_point[1]
            coordinates = [x,y]
            dataset.append(coordinates)

    elif shape_type == 4:
        shape = "circle"
        print(shape.upper())
        x, y = make_circles(n_samples=DATA_POINTS, shuffle=False, random_state=l)
        dataset = define_shape(x, y, train)
        dataset = dataset[:, [0, 1]]

    else:
        print("Select from 1 to 4")
        sys.exit(1)

    dataset = np.asarray(dataset)
    #print(dataset)
    #print(type(dataset))

    save_dataset(dataset, shape, train)
    return dataset


def save_dataset(data, shape, train=True):
    """

    :param data:
    :param shape:
    :return:
    """
    if train:
        label = 'train'
    else:
        label = 'test'

    filename = shape+'_'+ str(DATA_POINTS) + '_'+ label +".csv"
    print('[INFO] Saving generated dataset to', filename)
    np.savetxt('data/'+filename, data, delimiter=',')



if __name__ == "__main__":

    dataset = create_shape(4, train=False)
    utils.plot_trajectory(dataset)
