import torch

from torch import nn
from sklearn.utils import compute_class_weight
import pandas as pd
import numpy as np
import torch.nn.functional as F


class WeightedBCEwithlogitsLoss(nn.Module):
	def __init__(self,weight):
		super(WeightedBCEwithlogitsLoss, self).__init__()
		self.weights =
