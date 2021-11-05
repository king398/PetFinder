import numpy as np
import matplotlib.pyplot as plt

images = np.empty(shape=(30, 30, 3))
for i in range(10):
	x = np.random.randn(30, 30, 3)
	images = np.hstack((x, images))
images = np.array(images)
plt.imshow(images)
plt.show()
