import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2):
		super.__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, y_pred, y_true):
		ce = F.binary_cross_entropy(y_pred, y_true)
		pred_prob = y_pred
		p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
		alpha_factor = 1.0
		modulating_factor = 1.0
		if self.alpha:
			alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
		if self.gamma:
			modulating_factor = torch.pow((1.0 - p_t), self.gamma)

		return torch.sum(alpha_factor * modulating_factor * ce, dim=-1)


loss = FocalLoss()
pred = torch.rand(4)
true = torch.rand(4)
print(loss(pred, true))
