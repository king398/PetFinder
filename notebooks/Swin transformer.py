import pandas as pd, numpy as np, random, os, shutil
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import sklearn
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import imagesize
import wandb
import yaml

from vit_keras import vit
from IPython import display as ipd
from glob import glob
from tqdm.notebook import tqdm
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

print('np:', np.__version__)
print('pd:', pd.__version__)
print('sklearn:', sklearn.__version__)
print('tf:', tf.__version__)
print('tfa:', tfa.__version__)
print('w&b:', wandb.__version__)
import wandb

try:
	from kaggle_secrets import UserSecretsClient

	user_secrets = UserSecretsClient()
	api_key = user_secrets.get_secret("WANDB")
	wandb.login(key=api_key)
	anonymous = None
except:
	anonymous = "must"
	print(
		'To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')
