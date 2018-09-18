# EEGの時系列データをリアルタイムにとってるっぽくmatplotlibで表示するプログラム

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from mne_wrapper import get_raw

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
