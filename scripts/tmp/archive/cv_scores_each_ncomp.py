import numpy as np

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score
from sklearn import svm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mylib.mne_wrapper import get_cross_val_score


runs = [6, 10, 14]


def get_scores_each_ncomp():
    scores = []
    event_id = dict(hands=2, feet=3)
    for i in range(2, 21):
        score = get_cross_val_score(
            1, runs=runs, event_id=event_id, n_components=i, C=1.)
        scores.append(score)
        print(score)
    return scores


scores = get_scores_each_ncomp()
# print(scores)
