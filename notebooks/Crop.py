# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""

import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import glob
from backbone import EfficientDetBackbone
import cv2
import numpy as np
import matplotlib.pyplot as plt
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
	plot_one_box
from tqdm.notebook import tqdm

compound_coef = 6
force_input_size = None  # set None to use default size
img_paths = '../input/petfinder-pawpularity-score/test/*.jpg'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'../input/tak-efficientdet/efficientdet-d6.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
	model = model.cuda()
if use_float16:
	model = model.half()
for img_path in tqdm(glob.glob(img_paths)):
	color_list = standard_to_bgr(STANDARD_COLORS)
	# tf bilinear interpolation is different from any other's, just make do
	input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
	input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
	ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

	if use_cuda:
		x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
	else:
		x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

	x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

	with torch.no_grad():
		features, regression, classification, anchors = model(x)

		regressBoxes = BBoxTransform()
		clipBoxes = ClipBoxes()

		out = postprocess(x,
		                  anchors, regression, classification,
		                  regressBoxes, clipBoxes,
		                  threshold, iou_threshold)


	def display(preds, imgs):
		for i in range(len(imgs)):

			imgs[i] = imgs[i].copy()
			if "cat" not in list(preds[i]['class_ids']) or "dog" not in list(preds[i]['class_ids']):
				print()
				path = img_path.split("/")
				path = "./crop/" + path[4]
				cv2.imwrite(path, imgs[i])
			else:
				for j in range(len(preds[i]['rois'])):
					x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
					obj = obj_list[preds[i]['class_ids'][j]]
					if obj == "cat" or obj == "dog":
						imms = imgs[i]
						imms = imms[y1:y2, x1:x2]

						path = img_path.split("/")
						path = "./crop/" + path[4]
						cv2.imwrite(path, imms)
						break


	out = invert_affine(framed_metas, out)
	display(out, ori_imgs)
