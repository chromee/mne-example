# EEGの時系列データをリアルタイムにとってるっぽくmatplotlibで表示するプログラム

import os, sys, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.mne_wrapper import get_raw
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


raw = get_raw(subject=1, runs=[6, 10, 14], event_id=dict(hands=2, feet=3))
data = raw.get_data()

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

