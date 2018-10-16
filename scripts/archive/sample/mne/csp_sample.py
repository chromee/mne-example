import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne import Epochs, pick_types, find_events
import mne_wrapper

layout = read_layout('EEG1005')
subject = 1

epochs = mne_wrapper.get_epochs(subject)
epochs_data = epochs.get_data()
labels = epochs.events[:, -1] - 1


def check_csp(csp):
    x = csp.fit_transform(epochs_data, labels)
    print(x.shape)
    csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                      units='Patterns (AU)', size=1.5)
    csp.plot_filters(epochs.info, layout=layout, ch_type='eeg',
                     units='Patterns (AU)', size=1.5)


csp = CSP()
check_csp(csp)

csp = CSP(reg=0.5)
check_csp(csp)

csp = CSP(log=True)
check_csp(csp)

csp = CSP(cov_est="epoch")
check_csp(csp)

csp = CSP(transform_into="csp_space")
check_csp(csp)

csp = CSP(norm_trace=True)
check_csp(csp)
