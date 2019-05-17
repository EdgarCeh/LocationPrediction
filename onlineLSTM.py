"""
This is an "online" version of the LSTM for
next location prediction.
Online here means that the batch size is equal to 1.

This code will split the dataset in training and testing
Train the LSTM network using the training data.
Start testing with the testing data, but after
each prediction, updates the LSTM weights with
the instance used previously for prediction

"""
from Common import *
import Utilities as util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from torch.autograd import  Variable

TRAIN_LINE = 'data/line_20_train.csv'
DATA_POINTS = 20
#LSTM Hyper-parameters
seq_length = 3 #20
learning_rate = 0.01
features = 2 #14 #2      #input_size
hidden_size = 128 #50
output_size = 2     #output(X, Y)
num_layers = 2
num_epochs = 100#300


class LocationDataset(Dataset):
    """
    Class that generates the batches
    """
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = data.iloc[:, :-2]
        self.y_data = data.iloc[:, -2:]


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X = torch.from_numpy(np.asarray(self.x_data.iloc[idx]))
        Y = torch.from_numpy(np.asarray(self.y_data.iloc[idx]))

        return X, Y


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df = df.drop(0)
	return df


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)





def get_data(filename):
    """

    :param filename:
    :return:
    """
    df = pd.read_csv(filename, header=None, names=['X','Y'])
    data = df.values

    return data

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    agg.reset_index(drop=True, inplace=True)
    return agg


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("[INFO] Getting the data")
    # Get the data from file, and plot it
    dataset = get_data(TRAIN_LINE)
    ##util.plot_trajectory(dataset)
    # Scale the data in a range from -1 to 1, and plot it
    print("[INFO] Scaling the data")
    scaler, scaled_dataset = util.scale_data(dataset)
    ##util.plot_trajectory(scaled_dataset)

    print('[INFO] Generating sequences')
    #data = timeseries_to_supervised(dataset,1)
    scaled_data = series_to_supervised(scaled_dataset,seq_length,1)
    #print(scaled_data)
    #print(scaled_data.iloc[:,-2:])

    # Generate the sequences
    ##print('[INFO] Generate sequences')
    ##sequences, targets = util.sliding_windows(scaled_data, seq_length)
    ##print("\tSequences size:", sequences.shape)
    ##print("\tTargets size:", targets.shape)

    # Split the data in train-test
    print('[INFO] Splitting the data')
    train_size = int(len(scaled_data) * TRAIN_PCT)
    test_size = len(scaled_data) - train_size
    train = scaled_data.iloc[:train_size,:].reset_index(drop=True)
    test = scaled_data.iloc[train_size:,:].reset_index(drop=True)

    #print(train)
    #print(test)
    print("\tTrain dataset size:", train.shape[0])
    print("\tTest dataset size:", test.shape[0])

    #x_train = train.iloc[:, :-2]
    #y_train = train.iloc[:, -2:]
    #x_test = test.iloc[:, :-2]
    #y_test = test.iloc[:, -2:]

    #print(x_train)
    #print(np.asarray(x_train.iloc[4]))


    #train_X = sequences[0:train_size]
    ##test_X = sequences[train_size:len(sequences)]

    #train_Y = targets[0:train_size]
    ##test_Y = targets[train_size:len(targets)]

    # numpy.ndarray
    ##print("\tTrain X shape:", train_X.shape)
    #print('\t', type(train_X))
    ##print("\tTrain y shape:", train_Y.shape)
    #print('\t', type(train_Y))
    ##print("\tTest X shape:", test_X.shape)
    #print('\t', type(test_X))
    ##print("\tTest y shape:", test_Y.shape)
    #print('\t', type(test_Y))
    ##print()

    print("[INFO] Creating data loaders")
    train = LocationDataset(train)
    test = LocationDataset(test)

    trainLoader = DataLoader(dataset=train, batch_size=1, num_workers=2)
    testLoader = DataLoader(dataset=test, batch_size=1, num_workers=2)

    print("[INFO] Testing Train loader")
    for i, data in enumerate(trainLoader):
        inputs, target = data
        inputs = Variable(inputs)
        target = Variable(target)
        print("Inputs")
        print(inputs.numpy())
        print('Target')
        print(target.numpy())
        print('----')
        print()

    print()
    print("[INFO] Testing Test loader")
    for i, data in enumerate(testLoader):
        inputs, target = data
        inputs = Variable(inputs)
        target = Variable(target)
        print("Inputs")
        print(inputs.numpy())
        print('Target')
        print(target.numpy())
        print('----')
        print()
