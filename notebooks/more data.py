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
for i in ids:
	i.spilt(r"\\")