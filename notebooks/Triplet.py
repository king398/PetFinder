import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from timm.models.layers import get_act_layer


class BasicConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
	             bn=True, bias=False,
	             act_layer=nn.ReLU):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
		                      dilation=dilation, groups=groups, bias=bias)
		self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
		self.relu = act_layer if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x, inplace=True)
		return x


class ChannelPool(nn.Module):
	def forward(self, x):
		return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
	def __init__(self, act_layer=nn.ReLU, kernel_size=7):
		super(SpatialGate, self).__init__()
		self.compress = ChannelPool()
		self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False,
		                         act_layer=act_layer)

	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = torch.sigmoid_(x_out)
		return x * scale


class TripletAttention(nn.Module):
	def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False,
	             act_layer=nn.ReLU, kernel_size=7):
		super(TripletAttention, self).__init__()
		self.ChannelGateH = SpatialGate(act_layer=act_layer, kernel_size=kernel_size)
		self.ChannelGateW = SpatialGate(act_layer=act_layer, kernel_size=kernel_size)
		self.no_spatial = no_spatial
		if not no_spatial:
			self.SpatialGate = SpatialGate(kernel_size=kernel_size)

	def forward(self, x):
		x_perm1 = x.permute(0, 2, 1, 3).contiguous()
		x_out1 = self.ChannelGateH(x_perm1)
		x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
		x_perm2 = x.permute(0, 3, 2, 1).contiguous()
		x_out2 = self.ChannelGateW(x_perm2)
		x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
		if not self.no_spatial:
			x_out = self.SpatialGate(x)
			x_out = (1 / 3) * (x_out + x_out11 + x_out21)
		else:
			x_out = (1 / 2) * (x_out11 + x_out21)
		return x_out


class GeMP(nn.Module):
	def __init__(self, p=3., eps=1e-6, learn_p=False):
		super().__init__()
		self._p = p
		self._learn_p = learn_p
		self.p = nn.Parameter(torch.ones(1) * p)
		self.eps = eps
		self.set_learn_p(flag=learn_p)

	def set_learn_p(self, flag):
		self._learn_p = flag
		self.p.requires_grad = flag

	def forward(self, x):
		x = F.avg_pool2d(
			x.clamp(min=self.eps).pow(self.p),
			(x.size(-2), x.size(-1))
		).pow(1.0 / self.p)

		return x


class BlockAttentionModel(nn.Module):
	def __init__(
			self,
			backbone: nn.Module,
			n_features: int,
	):
		"""Initialize"""
		super(BlockAttentionModel, self).__init__()
		self.backbone = backbone
		self.n_features = n_features
		self.pooling = "gem"
		act_layer = nn.ReLU

		self.attention = TripletAttention(self.n_features,
		                                  act_layer=act_layer,
		                                  kernel_size=13)

		if self.pooling == 'avg':
			self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
		elif self.pooling == 'gem':
			self.global_pool = GeMP(p=4, learn_p=False)
		elif self.pooling == 'max':
			self.global_pool = torch.nn.AdaptiveMaxPool2d(1)
		elif self.pooling == 'nop':
			self.global_pool = torch.nn.Identity()
		else:
			raise NotImplementedError(f'Invalid pooling type: {self.pooling}')

		self.head = nn.Linear(self.n_features, 1)

	def _init_params(self):
		nn.init.xavier_normal_(self.fc.weight)
		if type(self.fc.bias) == torch.nn.parameter.Parameter:
			nn.init.constant_(self.fc.bias, 0)
		nn.init.constant_(self.bn.weight, 1)
		nn.init.constant_(self.bn.bias, 0)

	def forward(self, x, t=None):
		"""Forward"""
		x = self.backbone(x)
		print(x.shape)

		x = self.global_pool(x)
		x = x.view(x.size(0), -1)

		x = self.head(x)
		return x


backbone = timm.create_model("swin_large_patch4_window12_384", pretrained=True)
n_features = backbone.num_features
backbone.reset_classifier(0, '')
net = BlockAttentionModel(backbone, n_features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(net)
net = net.to(device)
input = torch.randn(1, 3, 384, 384)
input = input.to(device)
output = net(input)
print(output)
