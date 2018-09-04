# EEGの時系列データをリアルタイムにとってるっぽくmatplotlibで表示するプログラム

import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mylib.mne_wrapper import get_raw
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

subject = 1
raw = get_raw(subject)
data = raw.get_data()

sfreq = 40  # 本当は160Hzだけどなぜか時間が合うのは40
interval = 1. / sfreq
time_range = 10

ch = 1
fig, ax = plt.subplots(1, 1)
x = np.arange(0, time_range, interval)
y = data[ch][0:time_range*sfreq]
lines, = ax.plot(x, y)

for i in range(len(data[ch])):
    x += interval
    y = data[ch][i:i+time_range*sfreq]
    lines.set_data(x, y)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    plt.pause(interval)
