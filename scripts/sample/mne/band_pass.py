import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf

mne.set_log_level('WARNING')

def culc_by_csp_and_lda(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1).T
    ch_types = ["eeg" for i in range(16)] + ["stim"]
    ch_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", "O2", "STIM"]

    info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
    raw = mne.io.RawArray(data[1:], info)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    raw.plot(n_channels=len(ch_names), scalings='auto', title='Data from arrays', show=True, block=True)

# root = Path("./data/csv/judo")
# for path in root.iterdir():
#     if path.is_file():
#         culc_by_csp_and_lda(path)

path = Path("./data/csv/misl/afujii_MIK_20_07_2017_17_00_48_0000.csv")
culc_by_csp_and_lda(path)