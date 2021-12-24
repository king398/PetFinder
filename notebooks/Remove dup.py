
import os
import glob
import itertools
import collections

from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import imagehash

import matplotlib.pyplot as plt


def run():

    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]

    petids = []
    hashes = []
    for path in tqdm(glob.glob('E:\Pseudo Dogs/*.jpg')):

        image = Image.open(path)
        imageid = path.split('/')[-1].split('.')[0]

        petids.append(imageid)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))

    return petids, np.array(hashes)

petids, hashes_all = run()