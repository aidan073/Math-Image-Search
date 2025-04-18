import pandas as pd

df1 = pd.read_csv("mini_wiki/metadata.tsv", sep="\t", header=None)
df2 = pd.read_csv("Merged-Math-Dataset/Meta.tsv", sep="\t", header=None)
df1_filtered = df1[~df1[0].isin(df2[0])]
df1_filtered.to_csv("filtered_math_diff.tsv", sep="\t", header=False, index=False)