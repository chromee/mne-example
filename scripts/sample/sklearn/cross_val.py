
import numpy as np

x = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 4],
              [3, 4], [3, 4], [3, 4], [3, 4], [3, 4]])
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
print("KFold")
for train_index, test_index in kf.split(x, y):
    print("train_index:", train_index, "test_index:", test_index)


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)
print("StratifiedKFold")
for train_index, test_index in skf.split(x, y):
    print("train_index:", train_index, "test_index:", test_index)

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
print("ShuffleSplit")
for train_index, test_index in ss.split(x, y):
    print("train_index:", train_index, "test_index:", test_index)
