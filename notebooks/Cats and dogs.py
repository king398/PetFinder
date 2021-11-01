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


import albumentations
from albumentations.pytorch.transforms import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.to(device)

import cv2


def get_valid_transforms(DIM=384):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),
			albumentations.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
			ToTensorV2(p=1.0)
		]
	)


class CuteDataset(Dataset):
	def __init__(self, image_path, transform=get_valid_transforms(384)):
		self.image_path = image_path
		self.transform = transform

	def __len__(self):
		return len(self.image_path)

	def __getitem__(self, idx):
		print(idx)
		image_file = self.image_path[idx]
		image = cv2.imread(image_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.transform is not None:
			image = self.transform(image=image)['image']
		image_file = image_file.split("\\")
		image_file = str(image_file[6])
		image_file = image_file.split(".")
		if image_file[0] == "cat":
			target = 0
		else:
			target = 1
		label = torch.tensor(target).float()
		return image, label


train = CuteDataset(image_path=[r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train\cat.0.jpg"])
for image, label in train:
	print(label)
