import pandas as pd
df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\cat_class.csv")
second_row = df.loc[[1]]
print(second_row)