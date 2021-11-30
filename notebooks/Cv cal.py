import glob
import numpy as np
import matplotlib.pyplot as plt

cv = []

for i in glob.glob(r"D:\Models\SwinLarge2241kmixup/" + "*.pth"):
	i = i.split("_")
	print(i[8])
	cv.append(float(i[8]))
mean = sum(cv) / len(cv)

print(mean)

print(np.std(cv))
plt.plot(cv)
plt.show()

