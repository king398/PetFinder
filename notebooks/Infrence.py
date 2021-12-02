import sys

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
# Asthetics
import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import glob
import random
import cv2

pd.set_option('display.max_columns', None)

# Image Aug
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# Deep Learning
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Random Seed Initialize
RANDOM_SEED = 2021


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

csv_dir = '../input/petfinder-pawpularity-score'
if len(glob.glob("./crop/*.jpg")) == 0:
	test_dir = "../input/petfinder-pawpularity-score/test"
else:
	test_dir = "./crop"

models_dir = '../input/swin-transformenrs-pet-net'

test_file_path = os.path.join(csv_dir, 'test.csv')
sample_sub_file_path = os.path.join(csv_dir, 'sample_submission.csv')
print(f'Test file: {test_file_path}')
print(f'Models path: {models_dir}')

test_df = pd.read_csv(test_file_path)
sample_df = pd.read_csv(sample_sub_file_path)


def return_filpath(name, folder):
	path = os.path.join(folder, f'{name}.jpg')
	return path


test_df['image_path'] = test_df['Id'].apply(lambda x: return_filpath(x, folder=test_dir))
test_df.head()

df = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
print('Train shape:', df.shape)
df.head()

params = {
	'model': 'swin_large_patch4_window12_384_in22k',
	'model2': 'vit_large_patch16_224',
	'dense_features': ['Subject Focus', 'Eyes', 'Face', 'Near',
	                   'Action', 'Accessory', 'Group', 'Collage',
	                   'Human', 'Occlusion', 'Info', 'Blur'],
	'pretrained': False,
	'inp_channels': 3,
	'im_size': 384,
	'device': device,
	'batch_size': 8,
	'num_workers': 0,
	'out_features': 1,
	'debug': False
}

if params['debug']:
	test_df = test_df.sample(frac=0.1)


def get_test_transforms(DIM=params['im_size']):
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


class PetNet2(nn.Module):
	def __init__(self, model_name=params['model2'], out_features=params['out_features'],
	             inp_channels=params['inp_channels'],
	             pretrained=params['pretrained'], num_dense=len(params['dense_features'])):
		super().__init__()
		self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
		n_features = self.model.head.in_features
		self.model.head = nn.Linear(n_features, 128)
		self.fc = nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, out_features)
		)
		self.dropout = nn.Dropout(0.2)

	def forward(self, image, dense):
		embeddings = self.model(image)
		x = self.dropout(embeddings)
		x = torch.cat([x], dim=1)
		output = self.fc(x)
		return output


test_dataset = CuteDataset(
	images_filepaths=test_df['image_path'].values,
	dense_features=test_df[params['dense_features']].values,
	targets=sample_df['Pawpularity'].values,
	transform=get_test_transforms()
)
test_loader = DataLoader(
	test_dataset, batch_size=params['batch_size'],
	shuffle=False, num_workers=params['num_workers'],
	pin_memory=True
)

predicted_labels = None
for model_name in glob.glob("../input/d/mithilsalunkhe/swin-transformenrs-pet-net" + '/*.pth'):
	model = PetNet()
	model.load_state_dict(torch.load(model_name))
	model = model.to(params['device'])
	model.eval()

	test_dataset = CuteDataset(
		images_filepaths=test_df['image_path'].values,
		dense_features=test_df[params['dense_features']].values,
		targets=sample_df['Pawpularity'].values,
		transform=get_test_transforms()
	)
	test_loader = DataLoader(
		test_dataset, batch_size=params['batch_size'],
		shuffle=False, num_workers=params['num_workers'],
		pin_memory=True
	)

	temp_preds = None
	with torch.no_grad():
		for (images, dense, target) in tqdm(test_loader, desc=f'Predicting. '):
			images = images.to(params['device'], non_blocking=True)
			dense = dense.to(params['device'], non_blocking=True)
			predictions = torch.sigmoid(model(images, dense)).to('cpu').numpy() * 100

			if temp_preds is None:
				temp_preds = predictions
			else:
				temp_preds = np.vstack((temp_preds, predictions))

	if predicted_labels is None:
		predicted_labels = temp_preds
	else:
		predicted_labels += temp_preds

predicted_labels /= (len(glob.glob("../input/d/mithilsalunkhe/swin-transformenrs-pet-net" + '/*.pth')))

test_dataset = CuteDataset(
	images_filepaths=test_df['image_path'].values,
	dense_features=test_df[params['dense_features']].values,
	targets=sample_df['Pawpularity'].values,
	transform=get_test_transforms(224)
)

test_loader = DataLoader(
	test_dataset, batch_size=params['batch_size'],
	shuffle=False, num_workers=params['num_workers'],
	pin_memory=True
)

predicted_labels2 = None
for model_name in glob.glob("../input/vit-large-224/Vit1k" + '/*.pth'):
	model = PetNet2()
	model.load_state_dict(torch.load(model_name))
	model = model.to(params['device'])
	model.eval()

	temp_preds = None
	with torch.no_grad():
		for (images, dense, target) in tqdm(test_loader, desc=f'Predicting. '):
			images = images.to(params['device'], non_blocking=True)
			dense = dense.to(params['device'], non_blocking=True)
			predictions = torch.sigmoid(model(images, dense)).to('cpu').numpy() * 100

			if temp_preds is None:
				temp_preds = predictions
			else:
				temp_preds = np.vstack((temp_preds, predictions))

	if predicted_labels2 is None:
		predicted_labels2 = temp_preds
	else:
		predicted_labels2 += temp_preds

#     del model

predicted_labels2 /= (len(glob.glob("../input/vit-large-224/Vit1k" + '/*.pth')))
predicted_labels = predicted_labels * 0.6 + predicted_labels2 * 0.4
