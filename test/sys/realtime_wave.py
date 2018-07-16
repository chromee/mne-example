# EEGの時系列データをリアルタイムにとってるっぽくmatplotlibで表示するプログラム

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


path = "./data/afujii_MIK_20_07_2017_17_00_48_0000.csv"
data = np.loadtxt(path, delimiter=",", skiprows=1).T

sfreq = 512
interval = 1. / sfreq

fig, ax = plt.subplots(1, 1)
time_range = 1
x = np.arange(0, time_range, interval)
y = data[1][0:time_range*sfreq]
lines, = ax.plot(x, y)

for i in range(len(data[0])):
  # for ch in data:
  # print(data[1][i])
  x += interval
  y = data[1][i:i+time_range*sfreq]
  lines.set_data(x, y)
  ax.set_xlim((x.min(), x.max()))
  ax.set_ylim((y.min(), y.max()))
  plt.pause(interval)

