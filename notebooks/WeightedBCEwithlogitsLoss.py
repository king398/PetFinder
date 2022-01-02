import torch

from torch import nn

import numpy as np
import torch.nn.functional as F



class WeightedBCEwithlogitsLoss(nn.Module):
	def __init__(self, weight):
		super(WeightedBCEwithlogitsLoss, self).__init__()
		self.weights = torch.tensor(np.expand_dims(weight, 1)).cuda()

	def forward(self, output, target):

		loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
		for i, value in enumerate(loss):
			weights = int(target[i] * 100)  # get the index of values
			weightss = self.weights[weights - 1]
			loss[i] = loss[i] * weightss

		return loss.mean()
