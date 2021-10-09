#!/usr/bin/env python
# coding: utf-8

# ![SETI](https://www.petfinder.my/images/cuteness_meter.jpg)  
# 
# # Problem Statement
# * Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. You might expect pets with attractive photos to generate more interest and be adopted faster.
# * With the help of data science, we will accurately determine a pet photo’s appeal to give these rescue animals a higher chance of loving homes.
# * Currently, PetFinder.my uses a basic Cuteness Meter to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles.
# 
# ## Why this competition?
# As evident from the problem statement, this competition presents an interesting challenge for a good cause.  
# Also (if successful) the solution can be adapted into tools that will can shelters and rescuers around the world to improve the appeal of their pet profiles, automatically enhancing photo quality and consequently helping animals find a suitable hjome much faster.
# 
# ## Expected Outcome
# Given a photo a pet animal and some basic information about the photo as dense features, we should be able to estimate the 'pawpularity' score of the pet.
# 
# ## Data Description
# Image data is stored in a jpg image format in training folder and the dense features and target scores are mentioned in the `train.csv` file where the Id of each row corresponds to an unique image in the training folder.
# There are also some basic info on the photograph as dense features on the `train.csv` file.
# 
# ## Grading Metric
# Submissions are evaluated on **RMSE** between the predicted value and the observed target.
# 
# ## Problem Category
# From the data and objective its is evident that this is a **Regression Problem** in the Computer Vision domain.
# 
# **If you found this notebook useful and use parts of it in your work, please don't forget to show your appreciation by upvoting this kernel. That keeps me motivated and inspires me to write and share these public kernels.** 😊

# # About This Notebook:-
# * This notebook tried to demonstrate the use of Transfer learning using Pytorch and how to combine image features with dense features for various tasks.
# * We use a vanilla **vit_large_patch32_384** model for extracting image embeddings and concatenate them with the dense features on the last layer on a NN.
# * Refer [this link](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/275094) for description regarding using this particular methodology.
# * This notebook only covers the training part. Inference can be found in the notebook link below.
# 
# Inference Notebook:- https://www.kaggle.com/manabendrarout/transformers-classifier-method-starter-infer  
# 
# <p style='color: #fc0362; font-family: Segoe UI; font-size: 1.5em; font-weight: 300; font-size: 24px'>TLDR:- We treat this problem as a classification problem by scaling all targets between [0, 1] and use cross entropy loss as loss-function. It is known that transformer based models are performing better than classic CNN based models on this dataset.</p>

# # Get GPU Info

# In[1]:


# # Installations

# In[2]:


# # Imports

# In[ ]:


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

# In[ ]:


csv_dir = '../input/petfinder-pawpularity-score'
train_dir = r'F:\Pycharm_projects\PetFinder\data\train'
test_dir = r'F:\Pycharm_projects\PetFinder\data\test'

train_file_path = r"F:\Pycharm_projects\PetFinder\data\train_5folds.csv"
sample_sub_file_path = os.path.join("F:\Pycharm_projects\PetFinder\data\sample_submission.csv")

print(f'Train file: {train_file_path}')
print(f'Train file: {sample_sub_file_path}')

# In[ ]:


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(sample_sub_file_path)


# In[ ]:


def return_filpath(name, folder=train_dir):
	path = os.path.join(folder, f'{name}.jpg')
	return path


# In[ ]:


train_df['image_path'] = train_df['Id'].apply(lambda x: return_filpath(x))
test_df['image_path'] = test_df['Id'].apply(lambda x: return_filpath(x, folder=test_dir))

# In[ ]:


train_df.head()

# In[ ]:


test_df.head()

# In[ ]:


target = ['Pawpularity']
not_features = ['Id', 'kfold', 'image_path', 'Pawpularity']
cols = list(train_df.columns)
features = [feat for feat in cols if feat not in not_features]
print(features)

# # CFG

# In[ ]:


TRAIN_FOLDS = [0]

# In[ ]:


params = {
	'model': 'swin_large_patch4_window12_384',
	'dense_features': features,
	'pretrained': True,
	'inp_channels': 3,
	'im_size': 384,
	'device': device,
	'lr': 1e-5,
	'weight_decay': 1e-6,
	'batch_size': 4,
	'num_workers': 0,
	'epochs': 10,
	'out_features': 1,
	'dropout': 0.2,
	'num_fold': 5,
	'mixup': False,
	'mixup_alpha': 1.0,
	'scheduler_name': 'CosineAnnealingWarmRestarts',
	'T_0': 5,
	'T_max': 5,
	'T_mult': 1,
	'min_lr': 1e-7,
	'max_lr': 1e-4
}


# # Augmentations
# 
# There a well known concept called **image augmentations** in CNN. What augmentation generally does is, it artificially increases the dataset size by subtly modifying the existing images to create new ones (while training). One added advantage of this is:- The model becomes more generalized and focuses to finding features and representations rather than completely overfitting to the training data. It also sometimes helps the model train on more noisy data as compared to conventional methods.  
# 
# Example:-  
# ![](https://www.researchgate.net/publication/319413978/figure/fig2/AS:533727585333249@1504261980375/Data-augmentation-using-semantic-preserving-transformation-for-SBIR.png)  
# Source:- https://www.researchgate.net/publication/319413978/figure/fig2/AS:533727585333249@1504261980375/Data-augmentation-using-semantic-preserving-transformation-for-SBIR.png
# 
# One of the most popular image augmentation libraries is **Albumentations**. It has an extensive list of image augmentations, the full list can be found in their [documentation](https://albumentations.ai/docs/).  
# 
# *Tip:- Not all augmentations are applicable in all conditions. It really depends on the dataset and the problem. Example:- If your task is to identify if a person is standing or sleeping, applying a rotational augmentation can make the model worse.*  
# 
# With that in mind, let's define our augmentations:-

# ## 1. Train Augmentations

# In[ ]:


def get_train_transforms(DIM=params['im_size']):
	return albumentations.Compose(
		[
			albumentations.RandomResizedCrop(DIM, DIM),
			albumentations.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
			albumentations.HorizontalFlip(p=0.5),
			albumentations.VerticalFlip(p=0.5),

			ToTensorV2(p=1.0),
		]
	)


# ## 2. Mixup

# In[ ]:


def mixup_data(x, z, y, params):
	if params['mixup_alpha'] > 0:
		lam = np.random.beta(
			params['mixup_alpha'], params['mixup_alpha']
		)
	else:
		lam = 1

	batch_size = x.size()[0]
	if params['device'].type == 'cuda':
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	mixed_z = lam * z + (1 - lam) * z[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, mixed_z, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ## 3. Valid Augmentations

# In[ ]:


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


# # Dataset

# In[ ]:


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


# ## 1. Visualize Some Examples

# In[ ]:


X_train = train_df['image_path']
X_train_dense = train_df[params['dense_features']]
y_train = train_df['Pawpularity']
train_dataset = CuteDataset(
	images_filepaths=X_train.values,
	dense_features=X_train_dense.values,
	targets=y_train.values,
	transform=get_train_transforms()
)


# In[ ]:


def show_image(train_dataset=train_dataset, inline=4):
	plt.figure(figsize=(20, 10))
	for i in range(inline):
		rand = random.randint(0, len(train_dataset))
		image, dense, label = train_dataset[rand]
		plt.subplot(1, inline, i % inline + 1)
		plt.axis('off')
		plt.imshow(image.permute(2, 1, 0))
		plt.title(f'Pawpularity: {label}')


# In[ ]:


for i in range(3):
	show_image(inline=4)

# In[ ]:


del X_train, X_train_dense, y_train, train_dataset


# # Metrics

# In[ ]:


def usr_rmse_score(output, target):
	y_pred = torch.sigmoid(output).cpu()
	y_pred = y_pred.detach().numpy() * 100
	target = target.cpu() * 100

	return mean_squared_error(target, y_pred, squared=False)


# In[ ]:


class MetricMonitor:
	def __init__(self, float_precision=3):
		self.float_precision = float_precision
		self.reset()

	def reset(self):
		self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

	def update(self, metric_name, val):
		metric = self.metrics[metric_name]

		metric["val"] += val
		metric["count"] += 1
		metric["avg"] = metric["val"] / metric["count"]

	def __str__(self):
		return " | ".join(
			[
				"{metric_name}: {avg:.{float_precision}f}".format(
					metric_name=metric_name, avg=metric["avg"],
					float_precision=self.float_precision
				)
				for (metric_name, metric) in self.metrics.items()
			]
		)


# # Scheduler
# 
# Scheduler is essentially an function that changes our learning rate over epochs/steps. But why do we need to do that?
# 1. The first reason is that our network may become stuck in either saddle points or local minima, and the low learning rate may not be sufficient to break out of the area and descend into areas of the loss landscape with lower loss.
# 2. Secondly, our model and optimizer may be very sensitive to our initial learning rate choice. If we make a poor initial choice in learning rate, our model may be stuck from the very start.
# 
# Instead, we can use Schedulers and specifically Cyclical Learning Rates(CLR) to oscillate our learning rate between upper and lower bounds, enabling us to:
# * Have more freedom in our initial learning rate choices.
# * Break out of saddle points and local minima.
# 
# In practice, using CLRs leads to far fewer learning rate tuning experiments along with near identical accuracy to exhaustive hyperparameter tuning.

# In[ ]:


def get_scheduler(optimizer, scheduler_params=params):
	if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
		scheduler = CosineAnnealingWarmRestarts(
			optimizer,
			T_0=scheduler_params['T_0'],
			eta_min=scheduler_params['min_lr'],
			last_epoch=-1
		)

	elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=scheduler_params['T_max'],
			eta_min=scheduler_params['min_lr'],
			last_epoch=-1
		)
	return scheduler


# # CNN Model
# 
# We will inherit from the nn.Module class to define our model. This is a easy as well as effective way of defining the model as it allows very granular control over the complete NN. We are not using the full capability of it here since it is a starter model, but practicing similar definitions will help if/when you decide to play around a little more with the NN layers and functions.  
# 
# Also we are using timm for instancing a pre-trained model.  
# The complete list of Pytorch pre-trained image models through timm can be found [here](https://rwightman.github.io/pytorch-image-models/)  

# In[ ]:

class BasicConv(nn.Module):
	def __init__(
			self,
			in_planes,
			out_planes,
			kernel_size,
			stride=1,
			padding=0,
			dilation=1,
			groups=1,
			relu=True,
			bn=True,
			bias=False,
	):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv2d(
			in_planes,
			out_planes,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=groups,
			bias=bias,
		)
		self.bn = (
			nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
			if bn
			else None
		)
		self.relu = nn.ReLU() if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class ZPool(nn.Module):
	def forward(self, x):
		return torch.cat(
			(torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
		)


class AttentionGate(nn.Module):
	def __init__(self):
		super(AttentionGate, self).__init__()
		kernel_size = 7
		self.compress = ZPool()
		self.conv = BasicConv(
			2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
		)

	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.conv(x_compress)
		scale = torch.sigmoid_(x_out)
		return x * scale


class TripletAttention(nn.Module):
	def __init__(self, no_spatial=False):
		super(TripletAttention, self).__init__()
		self.cw = AttentionGate()
		self.hc = AttentionGate()
		self.no_spatial = no_spatial
		if not no_spatial:
			self.hw = AttentionGate()

	def forward(self, x):
		x_perm1 = x.permute(0, 2, 1, 3).contiguous()
		x_out1 = self.cw(x_perm1)
		x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
		x_perm2 = x.permute(0, 3, 2, 1).contiguous()
		x_out2 = self.hc(x_perm2)
		x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
		if not self.no_spatial:
			x_out = self.hw(x)
			x_out = 1 / 3 * (x_out + x_out11 + x_out21)
		else:
			x_out = 1 / 2 * (x_out11 + x_out21)
		return x_out


class PetNet(nn.Module):
	def __init__(self, model_name=params['model'], out_features=params['out_features'],
	             inp_channels=params['inp_channels'],
	             pretrained=params['pretrained'], num_dense=len(params['dense_features'])):
		super().__init__()
		self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
		self.base_model = nn.Sequential(*list(model.children())[:-2])
		n_features = self.model.head.in_features
		self.head = nn.Linear(512, self.cfg.target_size)

		self.attention = TripletAttention()
		self.fc = nn.Sequential(
			nn.Linear(128 + num_dense, 64),
			nn.ReLU(),
			nn.Linear(64, out_features)
		)
		self.dropout = nn.Dropout(params['dropout'])

	def forward(self, image, dense):
		x = self.base_model(image)
		x = self.attention(x)
		x = self.head(x)
		print(x.shape)
		x = torch.cat([x, dense], dim=1)


		output = self.fc(x)
		return output


# # Train and Validation Functions

# ## 1. Train Function

# In[ ]:


def train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler=None):
	metric_monitor = MetricMonitor()
	model.train()
	stream = tqdm(train_loader)

	for i, (images, dense, target) in enumerate(stream, start=1):
		if params['mixup']:
			images, dense, target_a, target_b, lam = mixup_data(images, dense, target.view(-1, 1), params)
			images = images.to(params['device'], dtype=torch.float)
			dense = dense.to(params['device'], dtype=torch.float)
			target_a = target_a.to(params['device'], dtype=torch.float)
			target_b = target_b.to(params['device'], dtype=torch.float)
		else:
			images = images.to(params['device'], non_blocking=True)
			dense = dense.to(params['device'], non_blocking=True)
			target = target.to(params['device'], non_blocking=True).float().view(-1, 1)

		output = model(images, dense)

		if params['mixup']:
			loss = mixup_criterion(criterion, output, target_a, target_b, lam)
		else:
			loss = criterion(output, target)

		rmse_score = usr_rmse_score(output, target)
		metric_monitor.update('Loss', loss.item())
		metric_monitor.update('RMSE', rmse_score)
		loss.backward()
		optimizer.step()

		if scheduler is not None:
			scheduler.step()

		optimizer.zero_grad()
		stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")


# ## 2. Validate Function

# In[ ]:


def validate_fn(val_loader, model, criterion, epoch, params):
	metric_monitor = MetricMonitor()
	model.eval()
	stream = tqdm(val_loader)
	final_targets = []
	final_outputs = []
	with torch.no_grad():
		for i, (images, dense, target) in enumerate(stream, start=1):
			images = images.to(params['device'], non_blocking=True)
			dense = dense.to(params['device'], non_blocking=True)
			target = target.to(params['device'], non_blocking=True).float().view(-1, 1)
			output = model(images, dense)
			loss = criterion(output, target)
			rmse_score = usr_rmse_score(output, target)
			metric_monitor.update('Loss', loss.item())
			metric_monitor.update('RMSE', rmse_score)
			stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

			targets = (target.detach().cpu().numpy() * 100).tolist()
			outputs = (torch.sigmoid(output).detach().cpu().numpy() * 100).tolist()

			final_targets.extend(targets)
			final_outputs.extend(outputs)
	return final_outputs, final_targets


# # Run

# In[ ]:


best_models_of_each_fold = []
rmse_tracker = []

# In[ ]:


for fold in TRAIN_FOLDS:
	print(''.join(['#'] * 50))
	print(f"{''.join(['='] * 15)} TRAINING FOLD: {fold + 1}/{train_df['kfold'].nunique()} {''.join(['='] * 15)}")
	# Data Split to train and Validation
	train = train_df[train_df['kfold'] != fold]
	valid = train_df[train_df['kfold'] == fold]

	X_train = train['image_path']
	X_train_dense = train[params['dense_features']]
	y_train = train['Pawpularity'] / 100
	X_valid = valid['image_path']
	X_valid_dense = valid[params['dense_features']]
	y_valid = valid['Pawpularity'] / 100

	# Pytorch Dataset Creation
	train_dataset = CuteDataset(
		images_filepaths=X_train.values,
		dense_features=X_train_dense.values,
		targets=y_train.values,
		transform=get_train_transforms()
	)

	valid_dataset = CuteDataset(
		images_filepaths=X_valid.values,
		dense_features=X_valid_dense.values,
		targets=y_valid.values,
		transform=get_valid_transforms()
	)

	# Pytorch Dataloader creation
	train_loader = DataLoader(
		train_dataset, batch_size=params['batch_size'], shuffle=True,
		num_workers=params['num_workers'], pin_memory=True
	)

	val_loader = DataLoader(
		valid_dataset, batch_size=params['batch_size'], shuffle=False,
		num_workers=params['num_workers'], pin_memory=True
	)

	# Model, cost function and optimizer instancing
	model = PetNet()
	model = model.to(params['device'])
	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
	                              weight_decay=params['weight_decay'],
	                              amsgrad=False)
	scheduler = get_scheduler(optimizer)

	# Training and Validation Loop
	best_rmse = np.inf
	best_epoch = np.inf
	best_model_name = None
	for epoch in range(1, params['epochs'] + 1):
		train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler)
		predictions, valid_targets = validate_fn(val_loader, model, criterion, epoch, params)
		rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)
		if rmse < best_rmse:
			best_rmse = rmse
			best_epoch = epoch
			if best_model_name is not None:
				os.remove(best_model_name)
			torch.save(model.state_dict(),
			           f"{params['model']}_{epoch}_epoch_f{fold + 1}_{rmse}_rmse.pth")
			best_model_name = f"{params['model']}_{epoch}_epoch_f{fold + 1}_{rmse}_rmse.pth"

	# Print summary of this fold
	print('')
	print(f'The best RMSE: {best_rmse} for fold {fold + 1} was achieved on epoch: {best_epoch}.')
	print(f'The Best saved model is: {best_model_name}')
	best_models_of_each_fold.append(best_model_name)
	rmse_tracker.append(best_rmse)
	print(''.join(['#'] * 50))
	del model
	gc.collect()
	torch.cuda.empty_cache()

print('')
print(f'Average RMSE of all folds: {round(np.mean(rmse_tracker), 4)}')

# In[ ]:


for i, name in enumerate(best_models_of_each_fold):
	print(f'Best model of fold {i + 1}: {name}')

# This is a simple starter kernel on implementation of Transfer Learning using Pytorch for this problem. Pytorch has many SOTA Image models which you can try out using the guidelines in this notebook.
# 
# I hope you have learnt something from this notebook. I have created this notebook as a baseline model, which you can easily fork and paly-around with to get much better results. I might update parts of it down the line when I get more GPU hours and some interesting ideas.
# 
# **If you liked this notebook and use parts of it in you code, please show some support by upvoting this kernel. It keeps me inspired to come-up with such starter kernels and share it with the community.**
# 
# Thanks and happy kaggling!
