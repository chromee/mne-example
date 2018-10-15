import glob
import pandas as pd

LR_flags = [1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2]
root = "C:/Users/dk334/Downloads/eeg_raw_data/csv/"
files = glob.glob(root+"*")
for file_path in files:
    print(file_path)

    ### テスト用ダミー
    # file_path = "C:/Users/dk334/Downloads/eeg_raw_data/afujii_MIK_20_07_2017_17_00_48_0000.mat.csv"
    # LR_flags = [1,1,2,1,2]
    # labels = [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1,0]

    df = pd.read_csv(file_path)
    labels = df["label"]

    LR_flag_index = 0
    for i in range(len(labels)):

        ### 0なら右も左もないので次へ
        if labels[i] == 0: 
            continue     

        if labels[i] != labels[i+1]:
            LR_flag_index += 1
            # print("flag", LR_flag_index)

        if LR_flag_index >= len(LR_flags):
            if LR_flags[LR_flag_index-1] == 2: labels[i] = 2
            break

        if LR_flags[LR_flag_index] == 2: labels[i] = 2
        # print(i, ":", LR_flags[LR_flag_index],":",labels[i])

    df.to_csv(file_path+"_cuntom.csv", index=False)
    
    