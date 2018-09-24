import numpy as np
import matplotlib.pyplot as plt

# a = np.array([1., 2, 6, 2, 1, 7])
# R = 3

# a1 = a.reshape(-1, R)
# print(a1)
# a2 = a.reshape(-1, R).mean(axis=1)
# print(a2)


input_arr = np.arange(161)
R = 20
split_arr = np.linspace(0, len(input_arr), num=R+1, dtype=int)
dwnsmpl_subarr = np.split(input_arr, split_arr[1:])
dwnsmpl_arr = np.array(list(np.mean(item) for item in dwnsmpl_subarr[:-1]))
print(split_arr.shape, dwnsmpl_arr.shape)
