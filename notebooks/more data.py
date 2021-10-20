import pandas as pd
import os
import glob
from tqdm import tqdm
import tensorflow as tf

df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_1.csv")
df1 = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train.csv")
df["AdoptionSpeed"] = 25 * (4 - df["AdoptionSpeed"].values)

pawpularity = pd.concat([df1["Pawpularity"], df["AdoptionSpeed"]])
ids = []
labels = []
import time

for i, x in tqdm(zip(df["PetID"].values, df["AdoptionSpeed"].values)):
	start = time.time()
	all_names = glob.glob(rf"F:\Pycharm_projects\PetFinder\data\train_old/{i}-*.jpg")
	end = time.time()
	print(end - start)
	start = time.time()
	for y in all_names:
		ids.append(y)
		labels.append(x)
	end = time.time()
	print(end - start)
