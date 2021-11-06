import glob
import numpy as np
import matplotlib.pyplot as plt

cv = []

for i in glob.glob(r"D:\Models\SwinEffnet/*.pth"):
	i = i.split("_")
	cv.append(float(i[9]))
mean = sum(cv) / len(cv)

print(mean)

print(np.std(cv))
plt.plot(cv)
plt.show()
