# coding: UTF-8
# https://hayataka2049.hatenablog.jp/entry/2018/02/22/234011

import numpy as np

from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold as SKF, GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support as prf

def main():
    rfc = RFC(n_estimators=100, n_jobs=-1)
    fs = SelectFromModel(rfc)
    pca = PCA()
    svm = SVC()

    # パイプライン化（特徴選択・次元削減・分類の組み立て）
    estimators = zip(["feature_selection", "pca", "svm"], [fs, pca, svm])
    pl = Pipeline(list(estimators))
    
    # パラメータチューニング
    parameters = {"feature_selection__threshold" : ["mean", "median"],
                  "pca__n_components" :[0.8, 0.5],
                  "svm__gamma" : [0.001, 0.01, 0.05],
                  "svm__C": [1, 10]}

    gclf = GridSearchCV(pl, parameters, n_jobs=-1, verbose=2)

    digits = load_digits()
    X = digits.data
    y = digits.target
    first_fold = True
    trues = []
    preds = []
    for train_index, test_index in SKF().split(X, y):
        if first_fold:
            gclf.fit(X[train_index], y[train_index])
            clf = gclf.best_estimator_
            first_fold = False
        clf.fit(X[train_index,], y[train_index])
        trues.append(y[test_index])
        preds.append(clf.predict(X[test_index]))

    true_labels = np.hstack(trues)
    pred_labels = np.hstack(preds)
    print("p:{0:.6f} r:{1:.6f} f1:{2:.6f}".format(
        *prf(true_labels, pred_labels, average="macro")))

if __name__ == "__main__":
    main()


