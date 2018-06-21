import numpy as np

import mne
from mne.datasets import sample
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

mne.set_log_level('WARNING')

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# (raw.copy().pick_types(meg='mag')
#            .del_proj(0)
#            .plot(duration=60, n_channels=100, remove_dc=False))

# raw.plot_psd(tmax=np.inf, fmax=250)

# # ECG
# average_ecg = create_ecg_epochs(raw).average()
# print('We found %i ECG events' % average_ecg.nave)
# joint_kwargs = dict(ts_args=dict(time_unit='s'),
#                     topomap_args=dict(time_unit='s'))
# average_ecg.plot_joint(**joint_kwargs)