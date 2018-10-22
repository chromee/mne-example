import pandas as pd


def find_best_score(path):
    book = pd.read_excel(path)
    id = book["mean_test_score"].idxmax()
    return book.iloc[id]


if __name__ == "__main__":
    serieses = []
    for i in range(1, 110):
        sbj = pd.Series(i, index=["subject"])
        info = find_best_score(
            "D:/OneDrive - 同志社大学/BCI/results/hands_vs_feet_three_class_grid_fft/grid_fft_%d.xlsx" % i)
        s = pd.concat([sbj, info])
        serieses.append(s)
    pd.DataFrame(serieses).to_excel(
        "data/best_scores.xlsx", index=False, header=True)
