import torch

from torch import nn
from sklearn.utils import compute_class_weight
import pandas as pd
import numpy as np
import torch.nn.functional as F

train_df = pd.read_csv(r"F:\Pycharm_projects\PetFinder\data\train_10folds.csv")
y_train = train_df['Pawpularity'].values

class_weights = compute_class_weight(
	'balanced',
	np.unique(y_train),
	y_train) # calculate the class weights
output = torch.randn((5))
print(output)  # most underrepresented class
target = torch.tensor([0.27, 0.27, 0.27, 0.27, 0.99]) #0.27 mst common weight and 0.99 the most uncommon weight
class_weights = np.expand_dims(class_weights, 1)
print(class_weights.shape)
weight = torch.tensor(class_weights)

criterion = nn.BCEWithLogitsLoss(reduction='none')

loss = criterion(output, target)
for i, value in enumerate(loss):
	weights = int(target[i] * 100)  # get the index of values
	weightss = weight[weights - 1]  # get the weights

	loss[i] = loss[i] * weightss  # multiply the loss by the weights
loss = loss.mean()

print(f"weighted loss {loss}")  #
criterion = nn.BCEWithLogitsLoss()
loss1 = criterion(output, target)
print(f" non weighted loss {loss1}")
