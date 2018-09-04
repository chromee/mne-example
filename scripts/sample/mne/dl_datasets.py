import mne
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

mne.set_log_level('WARNING')

runs = list(range(1,15))
for i in range(1,109):
    subject = i+1
    print(subject)
    raw_fnames = eegbci.load_data(subject, runs)
