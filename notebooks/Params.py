import timm
from torchsummary import summary
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

import torch
import torchvision
import timm
import torch.nn as nn

# Metrics

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
params = {
	'model_1': 'swin_large_patch4_window12_384_in22k',
	'model_2': 'tf_efficientnetv2_m_in21k',
	'pretrained': True,
	'inp_channels': 3,
	'im_size': 384,
	'device': device,
	'lr': 1e-5,
	'weight_decay': 1e-6,
	'batch_size': 2,
	'num_workers': 2,
	'epochs': 5,
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


class PetNet(nn.Module):
	def __init__(self, model_1_name=params['model_1'], model_2_name=params['model_2'],
	             out_features=params['out_features'], inp_channels=params['inp_channels'],
	             pretrained=params['pretrained']):
		super().__init__()

		# Transformer
		self.model_1 = timm.create_model(model_1_name, pretrained=pretrained,
		                                 in_chans=inp_channels)
		n_features_1 = self.model_1.head.in_features
		self.model_1.head = nn.Linear(n_features_1, 128)

		# Conventional CNN
		self.model_2 = timm.create_model(model_2_name, pretrained=pretrained,
		                                 in_chans=inp_channels)
		out_channels = self.model_2.conv_stem.out_channels
		kernel_size = self.model_2.conv_stem.kernel_size
		stride = self.model_2.conv_stem.stride
		padding = self.model_2.conv_stem.padding
		bias = self.model_2.conv_stem.bias
		self.model_2.conv_stem = nn.Conv2d(inp_channels, out_channels,
		                                   kernel_size=kernel_size,
		                                   stride=stride, padding=padding,
		                                   bias=bias)
		n_features_2 = self.model_2.classifier.in_features
		self.model_2.classifier = nn.Linear(n_features_2, 128)

		self.fc = nn.Sequential(
			nn.Linear(128 + 128, 64),
			nn.ReLU(),
			nn.Linear(64, out_features)
		)
		self.dropout = nn.Dropout(0.2)

	def forward(self, image):
		transformer_embeddings = self.model_1(image)
		conv_embeddings = self.model_2(image)
		features = torch.cat([transformer_embeddings, conv_embeddings],
		                     dim=1)
		x = self.dropout(features)
		output = self.fc(x)
		return output


model = PetNet()
model = model.to(params['device'])
summary(model, (3, 384, 384))
