import pandas as pd
import os
import glob
from tqdm import tqdm

df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_1.csv")
df1 = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train.csv")
df["AdoptionSpeed"] = 25 * (4 - df["AdoptionSpeed"].values)

pawpularity = pd.concat([df1["Pawpularity"], df["AdoptionSpeed"]])
ids = []
labels = []


def x():
	for i, x in tqdm(zip(df["PetID"].values, df["AdoptionSpeed"].values)):
		all_names = glob.glob(rf"F:\Pycharm_projects\PetFinder\data\train_old/{i}-*.jpg")
		for y in all_names:
			ids.append(y)
			labels.append(x)


from multiprocessing import Process

if __name__ == '__main__':
	p = Process(target=x, args=())
	p.start()
	p.join()
