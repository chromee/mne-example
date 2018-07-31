import numpy as np
import pylab as plt

import mne
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

import sys
sys.path.append("lib/SurfaceLaplacian/")
from surface_laplacian import surface_laplacian

mne.set_log_level('WARNING')

event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
raw = concatenate_raws(raw_files)

print(raw.info)
exit()

raw.rename_channels(lambda x: x.strip('.'))
# raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events = find_events(raw, shortest_event=0, stim_channel='STI 014')
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
montage = mne.channels.read_montage('biosemi64')
epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True, montage=montage)
print(montage)

surf_lap, surf_orig = surface_laplacian(epochs=epochs, m=4, leg_order=50, smoothing=1e-5, montage=montage)

bf_erp = surf_orig.average()
at_erp = surf_lap.average()

bf_erp.plot_topomap(np.arange(0, 0.7, 0.1), vmin=-10, vmax=10, units="$\mu V$", time_unit='ms', cmap="jet", title="Voltage", scalings=dict(eeg=1e6))
at_erp.plot_topomap(np.arange(0, 0.7, 0.1), vmin=-40, vmax=40, units="$\mu V/mm^2$", time_unit='ms', cmap="jet", title="Laplacian", scalings=dict(eeg=2e6))
plt.close()