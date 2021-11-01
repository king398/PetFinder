import albumentations
import PIL
import numpy as np
import matplotlib.pyplot as plt
import glob
import random


def get_train_transforms(DIM=1024):
	return albumentations.Compose(
		[
			albumentations.RandomResizedCrop(DIM, DIM),
			albumentations.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),

		]
	)


import cv2

paths = glob.glob(r"../input/petfinder-pawpularity-score/train/*.jpg")
for i in range(10):
	image_random = random.choice(paths)
	srcBGR = cv2.imread(image_random)
	image = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
	image = np.array(image)

	transform = get_train_transforms()
	image = transform(image=image)['image']
	plt.imshow(image)
	plt.show()
