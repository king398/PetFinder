import numpy as np
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import glob
import matplotlib.pyplot as plt
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from tqdm import tqdm

from sklearn.metrics import roc_auc_score


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.model = timm.create_model("vit_base_patch32_384", pretrained=True)
		n_features = self.model.head.in_features
		self.model.head = nn.Linear(n_features, 128)
		self.fc = nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.model(x)
		x = self.fc(x)
		return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()

model = Model()
model.to(device)


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


import PIL


class CuteDataset(Dataset):
	def __init__(self, image_path, transform=get_valid_transforms(384)):
		self.image_path = image_path
		self.transform = transform

	def __len__(self):
		return len(self.image_path)

	def __getitem__(self, idx):
		image_file = self.image_path[idx]
		image = PIL.Image.open(image_file)
		image = np.array(image)
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


x = glob.glob(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train\*.jpg")

shuffled = random.sample(x, len(x))
x_test = glob.glob(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train\*.jpg")

train_dl = CuteDataset(
	image_path=shuffled, transform=get_valid_transforms())

train_loader = DataLoader(train_dl, batch_size=24, shuffle=True, num_workers=0, pin_memory=True)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1
auc = []
for x in (range(epochs)):
	for i, (image, target) in tqdm(enumerate(train_loader, start=1)):
		loss = nn.BCELoss()

		optim.zero_grad()

		image = image.to(device)
		target = target.to(device)
		output = model(image)
		output = torch.squeeze(output)
		loss = loss(output, target)
		loss.backward()
		optim.step()
		auc.append(roc_auc_score(target.cpu().detach().numpy(), output.cpu().detach().numpy()))
		if i % 100 == 0:
			print(np.average(auc))
			print(loss)
	torch.save(model.state_dict(), "F:\Pycharm_projects\PetFinder\models\classfier.pth")
