import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from cv2 import imread
import cv2

start = timer()
x = imread(r"F:\Pycharm_projects\PetFinder\data\train\0a0da090aa9f0342444a7df4dc250c66.jpg")
x = cv2.resize(x, dsize=(768, 768))
plt.imshow(x)
plt.show()
x = torch.tensor(x)
x = torch.transpose(x, 0, 2)  # channels, height, width
kernel_size, stride = 384, 384
patches = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
patches = torch.transpose(patches, 0, 1)
print(patches.shape)
for i in patches:
	i = torch.transpose(i, 0, 2).numpy()
	plt.imshow(i)
	plt.show()
	print(i.shape)

print(patches.shape)
end = timer()
print(end - start)
