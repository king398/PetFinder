import pandas as pd

df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_1.csv")
df["AdoptionSpeed"] = df["AdoptionSpeed"].values * 25
df.to_csv("F:\Pycharm_projects\PetFinder\data\newdata.csv")
