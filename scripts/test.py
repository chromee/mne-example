from pathlib import Path
import numpy as np

# a=[1,2,3] + [4]
# print(a)

# dumy = Path("C:/Users/dk334/Downloads/data.csv")
# dumy.rename("aiueo.csv")

# a = [
#   [1,2,3],
#   [4,5,6]
# ]

# print(a[:])
# print(a[:][:])

# print(1, 2)

# s = np.arange(0, 10, 0.2)
# print(s)

# sw = np.zeros((5,5))
# sw = sw.reshape(1,1)
# print(sw[1:3,:0])
# sw += np.array([1,2])
# print(sw)

# l = list(range(1, 17))
# l.pop(0)
# l = [0] * 17
# print(l)

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from mylib.mne_wrapper import get_epochs


# epochs = get_epochs(1)
# epochs_data = epochs.get_data()
# labels = epochs.events[:, -1] - 2
# print(labels)

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
np.savetxt("test.csv", a.T, delimiter=",")
