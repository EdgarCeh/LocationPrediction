import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import  Variable
import sys

from Common import *
import Utilities as utils

df = utils.get_all_data(FILENAME_ALL_DATA)
df = utils.get_target_data(df, TARGET_ID)

x_data = [[x] for x in list(df[X].values)]
y_data = [[y] for y in list(df[Y].values)]

x_scale, x_data = utils.scale_data(x_data)
y_scale, y_data = utils.scale_data(y_data)

#utils.plot_trajectory(y_data)
utils.plot_trajectory((x_data, y_data))

sequences, targets = utils.sliding_windows(x_data, seq_length)
print("Sequence size:", sequences.shape)
print("target size:", targets.shape)
print(sequences[0])
print(targets[0])

