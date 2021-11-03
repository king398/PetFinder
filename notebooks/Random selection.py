import glob
from shutil import copyfile
import os
import random
from tqdm import tqdm
import shutil

x = os.listdir(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train")
print(os.path.join(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train", x[0]))
selected = random.choices(x, k=10000)
print(len(selected))
for i in tqdm(selected):
	try:
		path = os.path.join(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train", i)
		des = os.path.join(r"F:\Pycharm_projects\PetFinder\data\pseudo", i)
		shutil.move(path, des)
	except:
		pass
