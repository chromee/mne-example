import timeit
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import mne
from mne import Epochs, pick_types, find_events
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mylib.mne_wrapper import get_raw

subject = 1
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

csp = mne.decoding.CSP(reg=None, log=True, norm_trace=False)
svm = SVC()
pl = Pipeline([("csp", csp), ("svm", svm)])

params = {"csp__n_components": range(1, 6),
          "svm__C": [1, 5, 10, 20],
          "svm__gamma": [0.0001, 0.0005, 0.001, 0.01]}


def print_df(df):
    print(df[["param_csp__n_components",
              "param_svm__C", "param_svm__gamma",
              "mean_score_time",
              "mean_test_score"]])


def main1():
    clf = GridSearchCV(pl, params, n_jobs=-1)
    clf.fit(epochs_data_train, labels)
    df = pd.DataFrame(clf.cv_results_)
    print_df(df)
    df.to_excel("grid.xlsx", index=False, header=True)


if __name__ == "__main__":
    print(timeit.timeit(main1, number=1))
