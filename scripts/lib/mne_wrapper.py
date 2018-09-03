import numpy as np
from time import sleep

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import mne.decoding


mne.set_log_level('WARNING')

## how to import
# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from lib.mne_wrapper import get_epochs
# 
# get_epochs(subject=1)


# 10-20に変換 (T3 -> T7, T4 -> T8)
ch_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P3", "Pz", "P4", "O1", "O2", "STI 014"]
low_freq = 7.
high_freq = 30.


def get_epochs_from_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1).T
    ch_types = ["eeg" for i in range(16)] + ["stim"]

    info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
    raw = mne.io.RawArray(data[1:], info)
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')

    events = mne.find_events(raw, stim_channel='STIM')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    event_id = dict(right=1, left=2)
    epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
    return epochs


def get_raw(subject, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    raw_fnames = eegbci.load_data(subject, runs)
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
    raw = concatenate_raws(raw_files)
    raw.rename_channels(lambda x: x.strip('.'))
    raw.pick_channels(ch_names)
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    return raw


def get_epochs(subject, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    raw = get_raw(subject)
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
    # labels = epochs.events[:, -1] - 2
    return epochs


def stream_virtual_eeg_signal(subject, processing, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    raw = get_raw(subject)
    data = raw.get_data()
    sfreq = 160
    interval = 1. / sfreq
    for i in range(len(data[0])):
        processing(i)
        sleep(interval)

