import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

df_swin = pd.read_csv(r"F:\Pycharm_projects\PetFinder\oof files\Fastai\train_with_oof_swin_large_384_beta_0.5.csv")
df_vit = pd.read_csv(
	r"F:\Pycharm_projects\PetFinder\oof files\Fastai\train_with_oof_swin_large_224_beta_0.5.csv")
# train = df_swin.sample(n=1700, random_state=42).reset_index(drop=True)
true = df_swin["Pawpularity"].values
swin_pred = df_swin["oof"].values
swin_small_pred = df_vit["oof"].values
score = []
for ww in np.arange(0, 1.05, 0.05):
	oof3 = (1 - ww) * swin_pred + ww * swin_small_pred
	rsme = np.sqrt(np.mean((oof3 - true) ** 2.0))

	# print(f'{ww:0.2} CV Ensemble RSME =',rsme)
	score.append(rsme)
best_w = np.argmin(score) * 0.05
print(best_w)

print(mean_squared_error(true, swin_pred * 0.5 + swin_small_pred * 0.5, squared=False))
print(mean_squared_error(true,   swin_small_pred , squared=False))
print(mean_squared_error(true,   swin_pred , squared=False))
