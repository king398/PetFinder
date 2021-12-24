from shutil import copy
import glob
from tqdm import tqdm

images = glob.glob(r"E:\Dogs 2\*/*.jpg")
print(len(images))
for i in tqdm(images):
	copy(i, r"E:\Pseudo Dogs")
