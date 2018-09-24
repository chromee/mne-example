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
a = a.mean(axis=0)
print(a)
# np.savetxt("test.csv", a.T, delimiter=",")

# a = np.zeros((3, 4, 20))
# print(a)
# a = a.reshape((3, 80))
# print(a)
# a = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# np.savetxt("test.csv", a.T, delimiter=",")

# a = np.arange(0, 801/160, 1/160)
# print(a)

a = np.linspace(5., 25., 8)
b = list(zip(a[:-1], a[1:]))
# # [ 5. 7.85714286 10.71428571 13.57142857 16.42857143 19.28571429 22.14285714 25. ]
# print(a)
# # [ 5. 7.85714286 10.71428571 13.57142857 16.42857143 19.28571429 22.14285714]
# print(a[:-1])
# # [ 7.85714286 10.71428571 13.57142857 16.42857143 19.28571429 22.14285714 25. ]
# print(a[1:])
# # [(5.0, 7.857142857142858), (7.857142857142858, 10.714285714285715),
# # (10.714285714285715, 13.571428571428571), (13.571428571428571, 16.42857142857143),
# # (16.42857142857143, 19.285714285714285), (19.285714285714285, 22.142857142857142),
# # (22.142857142857142, 25.0)]
# print(b)
for freq, (fmin, fmax) in enumerate(b):
    print(freq, fmin, fmax)
