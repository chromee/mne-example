import numpy as np
import pandas as pd
import glob
from pathlib import Path

root = Path("C:/Users/dk334/workspace/mne-example/data")
for path in root.iterdir():

    # # index削除処理
    # data = pd.read_csv(file_path)
    # data = data.drop("i", axis=1)
    # data.to_csv(file_path+"_fix.csv", index=False)

    # # rename処理
    # new_name = path.name.replace(".mat.csv_cuntom.csv_fix", "")
    # new_path = root.joinpath(new_name)
    # path.rename(new_path)
    print("a")