import pandas as pd

df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_1.csv")
print(df["AdoptionSpeed"].values())
