import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

df_swin = pd.read_csv("F:\Pycharm_projects\PetFinder\oof files\swin_large_patch4_window12_384_in22k_oof.csv")
scores = []
for i in range(10):
	train = df_swin.sample(n=1700).reset_index(drop=True)
	true = train["true"].values
	swin_pred = train["pred"].values
	scores.append(mean_squared_error(true, swin_pred, squared=False))
print(f"std = {np.std(scores)}")
print(f"mean ={np.mean(scores)}")
print(f"best_cv = {min(scores)}")
print(f"worst_cv = {max(scores)}")
print(f"difference between worst and best cv {max(scores) - min(scores)}")
