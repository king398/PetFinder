import pandas as pd
import os
df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_1.csv")
df1 = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train.csv")
df["AdoptionSpeed"] = (df["AdoptionSpeed"].values * 25)
new = df["AdoptionSpeed"] + df1["Pawpularity"]

pawpularity = pd.concat([df1["Pawpularity"], df["AdoptionSpeed"]])
id = df1["Id"] + os.listdir(r"F:\Pycharm_projects\PetFinder\data\train_old")
print(new)
