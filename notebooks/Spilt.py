import pandas as pd

df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\cat_class.csv")
df = df[df.is_cat != 0]
df.to_csv(r"F:\Pycharm_projects\PetFinder\data\catonly.csv")
df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\cat_class.csv")
df = df[df.is_cat != 1]
df.to_csv(r"F:\Pycharm_projects\PetFinder\data\dogonly.csv")
