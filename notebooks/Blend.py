import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

df_swin = pd.read_csv("F:\Pycharm_projects\PetFinder\oof files\swin_base_patch4_window12_384_oof.csv")
df_vit = pd.read_csv(
	"F:\Pycharm_projects\PetFinder\oof files\swin_large_patch4_window7_224_oof.csv")

df_vit_1k = pd.read_csv(r"F:\Pycharm_projects\PetFinder\oof files\vit_large_patch16_224_oof.csv")
# train = df_swin.sample(n=1700, random_state=42).reset_index(drop=True)
true = df_swin["true"].values
swin_pred = df_swin["pred"].values
swin_small_pred = df_vit["pred"].values
vit_pred = df_vit_1k["pred"].values
score = []
for ww in np.arange(0, 1.05, 0.05):
	oof3 = (1 - ww) * swin_pred + ww * swin_small_pred
	rsme = np.sqrt(np.mean((oof3 - true) ** 2.0))
	print(rsme)
	print(ww)

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
print(mean_squared_error(true, swin_pred * 0.6 + swin_small_pred * 0.4, squared=False))
print(mean_squared_error(true, swin_small_pred, squared=False))
