import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne import Epochs, pick_types, find_events
import mne_wrapper

layout = read_layout('EEG1005')
subject = 1

epochs = mne_wrapper.get_epochs(subject)
epochs_data = epochs.get_data()
labels = epochs.events[:, -1]
freq = np.linspace(0, 160, 801)

# epochs.plot_psd()
# print(labels)


def plot_spectol(f, t, label):
    F = np.fft.fft(f)
    Amp = np.abs(F)

    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 17

    plt.subplot(121)
    plt.plot(t, f, label='f(n)')
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Signal", fontsize=20)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=25)
    leg.get_frame().set_alpha(1)

    plt.subplot(122)
    plt.plot(freq[0:int(len(freq)/2)], Amp[0:int(len(Amp)/2)], label='|F(k)|')
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Amplitude'+str(label), fontsize=20)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=25)
    leg.get_frame().set_alpha(1)

    plt.show()


def down_sample(x, R):
    split_arr = np.linspace(0, len(x), num=R+1, dtype=int)
    dwnsmpl_subarr = np.split(x, split_arr[1:])
    a = np.array(list(np.nanmean(item) for item in dwnsmpl_subarr[:-1]))
    return a


csp = CSP(n_components=4, reg=None, norm_trace=False,
          transform_into="csp_space")
x = csp.fit_transform(epochs_data, labels)

for i in range(10):
    # plot_spectol(x[i][1], epochs.times, labels[i])

    F = np.fft.fft(x[i][1])
    amp = np.abs(F)

    freq_2 = freq[0:int(len(freq)/2)]
    amp = amp[0:int(len(amp)/2)]

    n_sample = 100
    fr = down_sample(freq_2, n_sample)
    a = down_sample(amp, n_sample)

    print(i, amp.shape)
    plt.plot(fr, a, label='f(n)')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude'+str(labels[i]))
    plt.show()
