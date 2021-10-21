import pandas as pd
import os
import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np

df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_1.csv")
df1 = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train.csv")
df["AdoptionSpeed"] = 25 * (4 - df["AdoptionSpeed"].values)
ids = np.load("F:\Pycharm_projects\PetFinder\data\idold.npy")
labels = np.load("F:\Pycharm_projects\PetFinder\data\labels.npy")
pawpularity = np.concatenate([df1["Pawpularity"].values, labels])
ids_used = []
for i in tqdm(ids):
	i = i.split("\\")
	i = i[5]
	i = i.split(".")
	i = i[0]
	ids_used.append(i)
full_ids = list(df1["Id"].values) + ids_used
print(len(full_ids))
data = {"Id": full_ids, "Labels": pawpularity}
dict1 = pd.DataFrame.from_dict(data=data)
dict1.to_csv(r"F:\Pycharm_projects\PetFinder\data/moredata.csv")
