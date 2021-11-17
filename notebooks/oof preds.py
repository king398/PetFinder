# Asthetics
import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import random
import gc
import cv2
import glob
from sklearn.metrics import mean_squared_error

gc.enable()
pd.set_option('display.max_columns', None)

# Visialisation
import matplotlib.pyplot as plt

# Image Aug
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# Deep Learning
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Metrics
from sklearn.metrics import mean_squared_error

# Random Seed Initialize
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


seed_everything()

# Device Optimization
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

print(f'Using device: {device}')
train_dir = r"F:\Pycharm_projects\PetFinder\data\train"
test_dir = r'F:\Pycharm_projects\PetFinder\data\test'

train_file_path = r'F:\Pycharm_projects\PetFinder\data\train_10folds.csv'
sample_sub_file_path = r"F:\Pycharm_projects\PetFinder\data\sample_submission.csv"

print(f'Train file: {train_file_path}')
print(f'Train file: {sample_sub_file_path}')
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(sample_sub_file_path)


def return_filpath(name, folder=train_dir):
	path = os.path.join(folder, f'{name}.jpg')
	return path


train_df['image_path'] = train_df['Id'].apply(lambda x: return_filpath(x))
test_df['image_path'] = test_df['Id'].apply(lambda x: return_filpath(x, folder=test_dir))

params = {
	'model': 'swin_large_patch4_window12_384_in22k',
	'model_1': 'swin_large_patch4_window12_384_in22k',
	'model_2': 'tf_efficientnetv2_m_in21k',
	'dense_features': ['Subject Focus', 'Eyes', 'Face', 'Near',
	                   'Action', 'Accessory', 'Group', 'Collage',
	                   'Human', 'Occlusion', 'Info', 'Blur'],
	'pretrained': True,
	'inp_channels': 3,
	'im_size': 384,
	'device': device,
	'lr': 1e-5,
	'weight_decay': 1e-6,
	'batch_size': 16,
	'num_workers': 0,
	'epochs': 10,
	'out_features': 1,
	'dropout': 0.2,
	'num_fold': 10,
	'mixup': True,
	'mixup_alpha': 1.0,
	'scheduler_name': 'CosineAnnealingWarmRestarts',
	'T_0': 5,
	'T_max': 5,
	'T_mult': 1,
	'min_lr': 1e-7,
	'max_lr': 1e-4
}


def get_valid_transforms(DIM=params['im_size']):
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
	def __init__(self, images_filepaths, dense_features, targets, transform=None):
		self.images_filepaths = images_filepaths
		self.dense_features = dense_features
		self.targets = targets
		self.transform = transform

	def __len__(self):
		return len(self.images_filepaths)

	def __getitem__(self, idx):
		image_filepath = self.images_filepaths[idx]
		image = cv2.imread(image_filepath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if self.transform is not None:
			image = self.transform(image=image)['image']

		dense = self.dense_features[idx, :]
		label = torch.tensor(self.targets[idx]).float()
		return image, dense, label


class PetNet(nn.Module):
	def __init__(self, model_name=params['model'], out_features=params['out_features'],
	             inp_channels=params['inp_channels'],
	             pretrained=params['pretrained'], num_dense=len(params['dense_features'])):
		super().__init__()
		self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
		n_features = self.model.head.in_features
		self.model.head = nn.Linear(n_features, 128)
		self.fc = nn.Sequential(
			nn.Linear(128 + num_dense, 64),
			nn.ReLU(),
			nn.Linear(64, out_features)
		)
		self.dropout = nn.Dropout(0.2)

	def forward(self, image, dense):
		embeddings = self.model(image)
		x = self.dropout(embeddings)
		x = torch.cat([x, dense], dim=1)
		output = self.fc(x)
		return output


preds = []
true = []
for i in glob.glob(r"D:\Models/" + "*.pth"):
	fold = i.split('_')
	fold = fold[8]
	fold = list(fold)
	try:
		fold = int(fold[1] + fold[2])
	except:
		fold = int(fold[1])
	print(fold)
	valid = train_df[train_df['kfold'] == fold]
	model = PetNet()
	model.load_state_dict(torch.load(i))
	model.to(params["device"])

	model.eval()
	X_valid = valid['image_path']
	X_valid_dense = valid[params['dense_features']]
	y_valid = valid['Pawpularity'] / 100
	valid_dataset = CuteDataset(
		images_filepaths=X_valid.values,
		dense_features=X_valid_dense.values,
		targets=y_valid.values,
		transform=get_valid_transforms()
	)
	val_loader = DataLoader(
		valid_dataset, batch_size=params['batch_size'], shuffle=False,
		num_workers=params['num_workers'], pin_memory=True
	)
	with torch.no_grad():
		for (images, dense, target) in tqdm(val_loader, desc=f'Predicting. '):
			images = images.to(params['device'], non_blocking=True)
			dense = dense.to(params['device'], non_blocking=True)
			with torch.cuda.amp.autocast():
				predictions = torch.sigmoid(model(images, dense)).to('cpu').numpy() * 100
			target = target.to("cpu").numpy()
			predictions = np.squeeze(predictions)
			predictions = predictions.astype(np.float32)
			predictions = predictions.tolist()
			for i, x in zip(predictions, target):
				preds.append(i)
				true.append(x * 100)


print(mean_squared_error(true, preds, squared=False))
