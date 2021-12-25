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
	y_train)
batch_size = 8
nb_classes = 1
output = torch.tensor([0.99, 0.01, 0.98])  # most underrepresented class
target = torch.tensor([0.80, 0.3, 0.33])
class_weights = np.expand_dims(class_weights,1)
print(class_weights.shape)
weight = torch.tensor(class_weights)

criterion = nn.BCEWithLogitsLoss(reduction='none')




loss = criterion(output, target)
loss = loss * weight
loss = loss.mean()

print(f"weighted loss {loss}")  #
criterion = nn.BCEWithLogitsLoss()
loss = criterion(output, target)
print(f" non weighted loss {loss}")
