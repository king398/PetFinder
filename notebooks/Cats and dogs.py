import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs"
import timm

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(384),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(384),
                                      transforms.CenterCrop(384),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder('F:\Pycharm_projects\PetFinder\data\Cats And Dogs\PetImages', transform=train_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
model = timm.create_model("swin_large_patch4_window12_384_in22k", pretrained=True, num_classes=1)
model
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 512)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(512, 256)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(256, 2)),
    ('output', nn.LogSigmoid())
]))

model.classifier = classifier