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

TRAIN_LINE = 'data/line_1000_train.csv'# TRAIN_L #'data/line_10000_train.csv'
#DATA_POINTS = 50
#LSTM Hyper-parameters
seq_length = 5
learning_rate = 0.01
features = 2 #14 #2      #input_size
hidden_size = 1 #50
output_size = 2     #output(X, Y)
num_layers = 2
num_epochs = 3 #5

batch_size = 1      # how many instances at the time

updates = 3
seq_to_predict  = 5
iterations = 10  #3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

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


class LSTM(nn.Module):

    def __init__(self, output_size, features, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = output_size
        self.num_layers = num_layers
        self.input_size = features
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.5, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        ##print("x shape:",x.size())
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ##print("h_0 shape:",h_0.size())
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ##print("c_0 shape:", c_0.size())
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h_0, c_0)) #out: tensor of shape (batch, seq_length, hidden_size)
        #print(out.size())

        out = out[:, -1, :]
        #print(out.size())
        out = out.view(-1, hidden_size)
        #print(out.size())
        out = self.fc(out)
        #sys.exit(0)
        return out


def min_max_mean(all_predictions):
    """
    Calulate the min, max, and mean from the predictions for each time step
    :param all_predictions:
    :return:
    """
    all_means = []
    all_mins = []
    all_maxs = []
    all_stds = []

    if len(all_predictions.shape) == 2: # For the errors
        steps = all_predictions.shape[1]
        for i in range(steps):
            data = all_predictions[:, i]
            mean = np.mean(data, axis=0)
            min = np.amin(data, axis=0)
            max = np.amax(data, axis=0)
            std = np.std(data, axis=0)

            all_means.append(mean.tolist())
            all_mins.append(min.tolist())
            all_maxs.append(max.tolist())
            all_stds.append(std.tolist())

    else:
        steps = all_predictions.shape[1]    # For the Predictions

        for i in range(steps):
            data = all_predictions[:, i, :]
            mean = np.mean(data, axis=0)
            min = np.amin(data, axis=0)
            max = np.amax(data, axis=0)
            # Truncate decimals
            Ndecimals = 2
            helper = 10 ** Ndecimals
            mean = np.trunc(mean * helper) / helper
            min = np.trunc(min * helper) / helper
            max = np.trunc(max * helper) / helper
            all_means.append(mean.tolist())
            all_mins.append(min.tolist())
            all_maxs.append(max.tolist())

            std = np.std(data, axis=0)
            all_stds.append(std.tolist())

    return all_means, all_mins, all_maxs, all_stds


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
    print("[INFO] Sequence length:", seq_length)
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

    #########################
    #########################
    # Limit the test to a fixed number of predictions
    test = test.head(seq_to_predict)

    ##########################
    ##########################
    #print(train.head(3))
    #print(test.head(3))
    print("\tTrain dataset size:", train.shape[0])
    print("\tTest dataset size:", test.shape[0])

    print("[INFO] Creating data loaders")
    #train = LocationDataset(train)
    test = LocationDataset(test)

    #trainLoader = DataLoader(dataset=train, batch_size=batch_size, num_workers=2)
    testLoader = DataLoader(dataset=test, batch_size=batch_size, num_workers=2)

    lstm = LSTM(output_size, features, hidden_size, num_layers)
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    def concatenate_instances(df, instance, target):
        """

        :param df:
        :param tensor:
        :return: pandas df
        """
        instance = instance.numpy()
        #print(instance)
        instance = instance.reshape(1,-1)[0]
        #print(instance)

        target = target.numpy()
        #print(target)
        target = target.reshape(1, -1)[0]
        #print(target)

        row = np.concatenate((instance, target))
        #print(row)
        #print(row.shape)
        row = pd.Series(row, index=df.columns)
        #print(row)
        #print(type(row))
        #print("######")
        df = df.append(row,ignore_index=True)
        #print(df)

        return df

    def train_model(train, num_epochs=num_epochs):
        print("\tTrain dataset size:", train.shape[0])
        print("\tTotal epochs:", num_epochs)

        train = LocationDataset(train)
        trainLoader = DataLoader(dataset=train, batch_size=batch_size, num_workers=2)

        lstm.train()
        for epoch in range(num_epochs):
            for i, data in enumerate(trainLoader):
                #print("Loading:", i+1)
                inputs, target = data
                inputs = Variable(inputs).float()
                inputs = inputs.view(batch_size, seq_length, features)
                target = Variable(target).float()

                outputs = lstm(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                #print(outputs.size())
            print("\t\tEpoch %d, loss: %1.6f" % (epoch, loss.data))


    all_predictions = []
    all_r2_errors = []
    all_mse_errors = []



    for i in range(iterations):
        print("#" * 50)
        print("[INFO] Iteration:", i+1)
        print("[INFO] Training the model")
        ##if i == 0:
        ##    print("\tFirst time with initial data")
        train_model(train)

        train_copy = train.copy()
        print("[INFO] Online testing")
        # Testing
        all_targets = []
        predictions = []
        targets = []
        r2_errors = []
        mse_errors = []

        mean_predictions = []
        max_predictions = []
        low_predictions = []


        for j, data in enumerate(testLoader):
            lstm.eval()
            inputs, target = data
            inputs = Variable(inputs).float()
            inputs = inputs.view(batch_size, seq_length, features)
            target = Variable(target).float()
            test_predict = lstm(inputs)
            #print("Predicted:", test_predict.detach().numpy()[0])
            #print("Truth:", target[0].detach().numpy())

            yhat = test_predict.detach().numpy()[0]
            y = target[0].detach().numpy()

            yhat = util.inverse_scale_data(scaler, [yhat])
            y = util.inverse_scale_data(scaler, [y])

            # Truncate decimals
            Ndecimals = 2
            helper = 10 ** Ndecimals
            y = np.trunc(y * helper) / helper
            yhat = np.trunc(yhat * helper) / helper
            print("\t[predicted, target]:",yhat[0], y[0])

            predictions.append(yhat[0])
            targets.append(y[0])

            r2, mse = util.model_accuracy(y, yhat)
            #print("\tR2 error:", r2)
            #print("\tMSE error:", mse)
            r2_errors.append(r2)
            mse_errors.append(mse)

            new_train = concatenate_instances(train_copy, inputs, target) ######
            print(new_train.shape)

            #Train again but with this new target as train
            print("[INFO] Train considering previous instance")
            train_model(new_train, updates)
            train_copy  = new_train ######

        #train = train_copy.copy()
        #del train_copy

        all_predictions.append(predictions)
        all_targets.append(targets)
        all_r2_errors.append(r2_errors)
        all_mse_errors.append(mse_errors)


    all_predictions =np.asarray(all_predictions)
    #print(all_predictions)
    #print(all_predictions.shape)
    #print()
    all_targets = np.asarray(all_targets)
    #print(all_targets)
    #print(all_targets.shape)
    print()

    print("~~~Results~~~")
    all_means, all_mins, all_maxs, all_stds = min_max_mean(all_predictions)
    #print(all_means)
    print("Min values:", all_mins)
    print("Max values:",all_maxs)
    print("Std deviation:", all_stds)
    predictions = np.asarray(all_means)
    #predictions = np.asarray(predictions)
    ##print(predictions)
    #print(type(predictions))

    #print("Targets"
    targets = np.asarray(targets)
    ##print(targets)
    #print(type(targets))

    print("Mean values", predictions.tolist())
    print("Targets    ", targets.tolist())

    print()
    print("R2 Errors")
    all_r2_errors = np.asarray(all_r2_errors)
    #print(all_r2_errors)
    #print(all_r2_errors.shape)
    r2_errors, _, _, _ = min_max_mean(all_r2_errors)
    print(r2_errors)

    print("MSE Errors")
    all_mse_errors = np.asarray(all_mse_errors)
    #print(all_mse_errors)
    #print(all_mse_errors.shape)
    mse_errors, _, _ , _ = min_max_mean(all_mse_errors)
    print(mse_errors)

    util.plot_compre_trajectories(targets, predictions)

    sys.exit(0)
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
