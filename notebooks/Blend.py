import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

df_swin = pd.read_csv("F:\Pycharm_projects\PetFinder\oof files\swin_large_patch4_window12_384_in22k_oof.csv")
df_vit = pd.read_csv(r"F:\Pycharm_projects\PetFinder\oof files\vit_large_patch16_224_in21k_oof.csv")
#train = df_swin.sample(n=1700, random_state=42).reset_index(drop=True)
true = df_swin["TRUE"].values
swin_pred = df_swin["pred"].values
print(mean_squared_error(true, swin_pred, squared=False))
