import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

images = torch.rand(1, 3, 384, 384)


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.model = timm.create_model("swin_large_patch4_window12_384_in22k", pretrained=True)
		n_features = self.model.head.in_features
		self.model.head = nn.Linear(n_features, 128)
		self.fc = nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

	def forward(self, x):
		x = self.model(x)
		print(x)
		x = self.fc(x)
		return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.to(device)

import cv2


class CuteDataset(Dataset):
	def __init__(self, image_path):
		self.image_path = image_path

	def __len__(self):
		return len(self.image_path)

	def __getitem__(self, idx):
		image_file = self.image_path[idx]
		image = cv2.imread(image_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print(image_file.spilt("//"))
		image = torch.tensor(image)
		return image


train = CuteDataset(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train\cat.0.jpg")
for i in train:
	print(train)
