import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import tfr_morlet, psd_multitaper

mne.set_log_level('WARNING')

tmin, tmax = -1., 4.
subject = 2
event_id = dict(hands=2, feet=3)
runs = [6, 10, 14]  # motor imagery: hands vs feet
ch_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3",
            "Cz", "C4", "T8", "P3", "Pz", "P4", "O1", "O2", "STI 014"]

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)
raw.rename_channels(lambda x: x.strip('.'))
raw.pick_channels(ch_names)
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
events = find_events(raw, shortest_event=0, stim_channel='STI 014')
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

# mne fft
# epochs.plot_psd()

epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1] - 2

epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

f = np.fft.fft(epochs_data[0][0])
Amp = np.abs(f)

plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# plt.subplot(121)
# t = np.arange(0, 801/160, 1/160)
# plt.plot(t, epochs_data[0][0], label='f(n)')
# plt.xlabel("Time", fontsize=20)
# plt.ylabel("Signal", fontsize=20)
# plt.grid()
# leg = plt.legend(loc=1, fontsize=25)
# leg.get_frame().set_alpha(1)

plt.subplot(121)
freq = np.linspace(0, 160, 801)
plt.plot(freq, Amp, label='|F(k)|')
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.grid()
leg = plt.legend(loc=1, fontsize=25)
leg.get_frame().set_alpha(1)
plt.show()

csp = CSP(n_components=16, reg='ledoit_wolf', norm_trace=False)
csp.fit_transform(epochs_data, labels)
print(csp.patterns_.shape)

# layout = read_layout('EEG1005')
# csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
#                   units='Patterns (AU)', size=1.5)

# f = np.fft.fft(x[0][0])
# Amp = np.abs(f)
# plt.subplot(122)
# freq = np.linspace(0, 160, 801)
# plt.plot(freq, Amp, label='|F(k)|')
# plt.xlabel('Frequency', fontsize=20)
# plt.ylabel('Amplitude', fontsize=20)
# plt.grid()
# leg = plt.legend(loc=1, fontsize=25)
# leg.get_frame().set_alpha(1)
# plt.show()
