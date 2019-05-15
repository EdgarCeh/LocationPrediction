"""
This LSTM model works with only one target, but using all data columns
The target needs to be specified in Common.TARGET_ID
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import  Variable
import sys

import pandas as pd

from Common import *
import Utilities as utils

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

df = utils.get_all_data(FILENAME_ALL_DATA)
df = utils.get_target_data(df, TARGET_ID)

print(df.shape)

if CONCAT :
    df = pd.concat([df]*N_TIMES)

print(df.head(seq_length+1))
print(df.shape)
columns = list(df.columns)[1:]      # Remove "id" column
print(columns)
print("Total columns:", len(columns))

data = df[columns].values.tolist()
print(data)
print(np.asarray(data).shape)
sc, scaled_data = utils.scale_data(data)
print(scaled_data[:seq_length+1])
print(np.asarray(scaled_data).shape)
print(type(scaled_data))

# The second parameter is the pair of columns to plot
#  [11, 12] =  XPixel, YPixel
#  [14, 15] = XLocation, YLocation
# Default uses [0,1]
utils.plot_trajectory(scaled_data, [14,15])

#
sequences, targets = utils.sliding_windows_all_data(scaled_data, seq_length)
print("Sequences size:", sequences.shape)
print("Targets size:", targets.shape)

print("Sequences:")
print(sequences[0])
print("Targets:")
print(targets[0])

##real_seq = utils.inverse_scale_data(sc, sequences[0])
##print(real_seq)
##real_target = utils.inverse_scale_data(sc, [targets[0]])
##print(real_target)

train_size = int(len(targets) * TRAIN_PCT)
test_size = len(targets) - train_size
print("Train size:", train_size)
print("Test size:", test_size)

train_X = sequences[0:train_size]
test_X = sequences[train_size:len(sequences)]

train_Y = targets[0:train_size]
test_Y = targets[train_size:len(targets)]

trainX = torch.Tensor(train_X)
trainX = Variable(trainX).to(device)

testX = torch.Tensor(test_X)
testX = Variable(testX).to(device)

trainY = torch.Tensor(train_Y)
trainY = Variable(trainY).to(device)

testY = torch.Tensor(test_Y)
testY = Variable(testY).to(device)

print("Train X size:", trainX.size())
print("Train y size:", trainY.size())

print("Test X size:", testX.size())
print("Test Y size:", testX.size())


class LSTM(nn.Module):

    def __init__(self, output_size, features, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = output_size
        self.num_layers = num_layers
        self.input_size = features
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        ##print("h_0 shape:",h_0.size())
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
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


lstm = LSTM(output_size, features, hidden_size, num_layers).to(device)
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    #print(outputs)
    #print(outputs.size())
    #print(train_Y)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    print("Epoch: %d, loss: %1.5f" % (epoch, loss.data))


lstm.eval()
test_predict = lstm(testX)

##test_predict = utils.inverse_scale_data(sc, test_predict.data.cpu().numpy())
##testY = utils.inverse_scale_data(sc, testY.data.cpu().numpy())

test_predict = test_predict.data.cpu().numpy()
testY = testY.data.cpu().numpy()


test_acc = utils.model_accuracy(testY, test_predict)

print(testY.shape)
print(test_predict.shape)

print("Test Accuracy (MSE): ", test_acc)

utils.plot_compre_trajectories(testY, test_predict)
