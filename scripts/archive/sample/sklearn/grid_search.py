import timeit
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

digits = load_digits()
svm = SVC()
pca = PCA(svd_solver="randomized")
pl = Pipeline([("pca", pca), ("svm", svm)])

params = {"pca__n_components": [10, 15, 30, 45],
          "svm__C": [1, 5, 10, 20],
          "svm__gamma": [0.0001, 0.0005, 0.001, 0.01]}


def print_df(df):
    print(df[["param_pca__n_components",
              "param_svm__C", "param_svm__gamma",
              "mean_score_time",
              "mean_test_score"]])


def main1():
    clf = GridSearchCV(pl, params, n_jobs=-1)
    clf.fit(digits.data, digits.target)
    df = pd.DataFrame(clf.cv_results_)
    print_df(df)


if __name__ == "__main__":
    print(timeit.timeit(main1, number=1))
