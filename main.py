import pandas as pd

train = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train.csv")
sumofvalues = []
pawpularity = []
for i in range(10):
	x1 = train.loc[i, :].values.tolist()
	x1.pop(0)
	pawpularity.append(int(x1[12]))
	x1.pop(12)
	sumofvalues.append(sum(list(x1)))
print(sumofvalues)
print(pawpularity)
