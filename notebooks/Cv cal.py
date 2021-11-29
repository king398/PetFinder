import glob
import numpy as np
import matplotlib.pyplot as plt

cv = []

for i in glob.glob(r"D:\Models\Vit1k/" + "*.pth"):
	i = i.split("_")
	cv.append(float(i[7]))
mean = sum(cv) / len(cv)

print(mean)

print(np.std(cv))
plt.plot(cv)
plt.show()
import glob
import numpy as np
import matplotlib.pyplot as plt

cv = []

for i in glob.glob(r"D:\Models\SwinBase1k/" + "*.pth"):
	i = i.split("_")
	cv.append(float(i[8]))
mean = sum(cv) / len(cv)

print(mean)

print(np.std(cv))
plt.plot(cv)
plt.show()
