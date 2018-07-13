import numpy as np
import pandas as pd
import glob

root = "C:/Users/dk334/workspace/mne-example/data/"
file_paths = glob.glob(root+"*")
for file_path in file_paths:
    # # index削除処理
    # data = pd.read_csv(file_path)
    # data = data.drop("i", axis=1)
    # data.to_csv(file_path+"_fix.csv", index=False)
