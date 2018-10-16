import pandas as pd
from pathlib import Path

p = Path("data/riemann")
df = pd.DataFrame()

for path in p.iterdir():
    tmp_df = pd.read_excel(path)
    df = pd.concat([df, tmp_df])

df.to_excel("data/two_class_riemann.xlsx", index=False)
