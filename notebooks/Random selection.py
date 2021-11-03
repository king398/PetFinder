import glob
from shutil import copyfile
import os
import random

x = os.listdir(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train")
print(os.path.join(r"F:\Pycharm_projects\PetFinder\data\Cats And Dogs\train", x[0]))
selected = random.choices(x, k=10000)
print(len(selected))
