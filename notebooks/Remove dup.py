import hashlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import numpy as np

from matplotlib.pyplot import imread, imshow
from tqdm import tqdm
import os

os.getcwd()
os.chdir(r'E:\Pseudo Dogs')
os.getcwd()
file_list = os.listdir()
print(len(file_list))
import hashlib, os

duplicates = []
hash_keys = dict()
for index, filename in tqdm(enumerate(os.listdir('.'))):  # listdir('.') = current directory
	if os.path.isfile(filename):
		with open(filename, 'rb') as f:
			filehash = hashlib.md5(f.read()).hexdigest()
		if filehash not in hash_keys:
			hash_keys[filehash] = index
		else:
			duplicates.append((index, hash_keys[filehash]))
print(len(duplicates))
for index in duplicates:
	os.remove(file_list[index[0]])
