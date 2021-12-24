import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

df_swin = pd.read_csv("F:\Pycharm_projects\PetFinder\oof files\swin_large_patch4_window12_384_oof_crop.csv")
df_vit = pd.read_csv(
	"F:\Pycharm_projects\PetFinder\oof files\swin_large_patch4_window7_224_oof_crop.csv")
df_swin_22k = pd.read_csv(r"F:\Pycharm_projects\PetFinder\oof files\swin_base_patch4_window12_384_in22k_oof.csv")
# train = df_swin.sample(n=1700, random_state=42).reset_index(drop=True)
true = df_swin["true"].values
swin_pred = df_swin["pred"].values
swin_small_pred = df_vit["pred"].values
swin_22k_pred = df_swin_22k["pred"].values
score = []
for ww in np.arange(0, 1.05, 0.05):
	oof3 = (1 - ww) * swin_pred + ww * swin_small_pred
	rsme = np.sqrt(np.mean((oof3 - true) ** 2.0))

	# print(f'{ww:0.2} CV Ensemble RSME =',rsme)
	score.append(rsme)
best_w = np.argmin(score) * 0.05
print(best_w)
plt.figure(figsize=(20, 5))
plt.plot(np.arange(21) / 20.0, score, '-o')
plt.plot([best_w], np.min(score), 'o', color='black', markersize=15)
plt.title(f'Best Overall CV RSME={np.min(score):.4} with SVR Ensemble Weight={best_w:.2}', size=16)
plt.ylabel('Overall Ensemble RSME', size=14)
plt.xlabel('SVR Weight', size=14)
plt.show()
print(mean_squared_error(true, swin_pred * 1/3 + swin_small_pred * 1/3 + swin_22k_pred * 1/3, squared=False))
