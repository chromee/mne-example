import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import mlab

n = 1024*4
fs = 512
dt = 1/fs

b = signal.firwin(30, 0.2, window="boxcar")
x = np.random.randn(n)
y = signal.filtfilt(b, 1, x)

w, h = signal.freqz(b, 1, fs)

plt.plot(w*fs/(2*np.pi), 20*np.log10(np.abs(h)), "b")
plt.xlim(0, fs/2)
plt.ylim(-120, 0)
plt.xlabel("Frequency[Hz]")
plt.ylabel("Amplitude[dB]")
plt.show()
