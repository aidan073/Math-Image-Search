import pandas as pd

"""
Check which samples from dataset 1 are not in dataset 2. Used to check which samples were filtered.
"""

df1 = pd.read_csv("splits/train_split.tsv", sep="\t", header=None)
df2 = pd.read_csv("Sim-0.7-Train/Meta.tsv", sep="\t", header=None)
df1_filtered = df1[~df1[0].isin(df2[0])]
df1_filtered.to_csv("filtered_by_sim.tsv", sep="\t", header=False, index=False)