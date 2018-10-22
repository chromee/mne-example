import pandas as pd

svm_df = pd.read_excel("data/three_class_svm.xlsx")
riemann_df = pd.read_excel("data/three_class_riemann.xlsx")
columns = ["subject", "svm_mean_test_score", "svm_mean_train_score",
           "riemann_mean_test_score", "riemann_mean_train_score"]
results = []
for i in range(109):
    print(i)
    df = pd.DataFrame([[i+1, svm_df["mean_test_score"][i], svm_df["mean_train_score"][i],
                        riemann_df["mean_test_score"][i], riemann_df["mean_train_score"][i]]], columns=columns)
    results.append(df)
df = pd.concat(results)
df.to_excel("data/svm_vs_riemann.xlsx", index=False)
