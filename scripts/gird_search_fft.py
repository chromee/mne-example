import timeit
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import mne
from mne import Epochs, pick_types, find_events
from mne_wrapper import get_raw, FFT


def grid_search(subject=1):
    runs = [6, 10, 14]
    event_id = dict(rest=1, hands=2, feet=3)
    tmin, tmax = -1., 4.

    raw = get_raw(subject, runs)
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_data_train = epochs.copy().crop(tmin=1., tmax=2.).get_data()
    labels = epochs.events[:, -1] - 1

    csp = mne.decoding.CSP(reg=None,
                           norm_trace=False, transform_into="csp_space")
    fft = FFT()
    svm = SVC()
    pl = Pipeline([("csp", csp), ("fft", fft), ("svm", svm)])

    params = {"csp__n_components": np.arange(4, 10),
              "svm__C": np.logspace(-4, 0, 5),
              "svm__gamma": np.logspace(-4, 0, 5)}

    clf = GridSearchCV(pl, params, n_jobs=-1)
    clf.fit(epochs_data_train, labels)
    df = pd.DataFrame(clf.cv_results_)
    # print(df[["param_csp__n_components",
    #           "param_svm__C", "param_svm__gamma",
    #           "mean_score_time",
    #           "mean_test_score"]])
    df.to_excel("grid_%s.xlsx" % subject, index=False, header=True)
    print("%s end" % subject)


if __name__ == "__main__":
    grid_search(1)
