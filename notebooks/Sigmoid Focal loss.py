import torch
from torch.nn.functional import log_softmax
from torch import nn


class SigmoidFocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2):
		super(SigmoidFocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, input, target):
		num_classes = input.shape[1]
		dtype = target.dtype
		device = target.device
		class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)
		t = target.unsqueeze(1)
		p = torch.sigmoid(input)
		term1 = (1 - p) ** self.gamma * torch.log(p)
		term2 = p ** self.gamma * torch.log(1 - p)
		return torch.mean(
			-(t == class_range).float() * term1 * self.alpha - ((t != class_range) * (t >= 0)).float() * term2 * (
					1 - self.alpha))


cri = SigmoidFocalLoss()
input = torch.randn(1, 3)
target = torch.randn(1, 3)
print(cri(input, target))


class FocalLoss(nn.Module):
	"""
	binary focal loss
	"""

	def __init__(self, alpha=0.25, gamma=2):
		super(FocalLoss, self).__init__()
		self.weight = torch.Tensor([alpha, 1 - alpha])
		self.nllLoss = nn.BCEWithLogitsLoss(weight=self.weight)
		self.gamma = gamma

	def forward(self, input, target):
		softmax = torch.nn.functional.sigmoid(input)
		log_logits = torch.log(softmax)
		fix_weights = (1 - softmax) ** self.gamma
		logits = fix_weights * log_logits
		return self.nllLoss(logits, target)


cri2 = FocalLoss()
print(cri2(input, target))
