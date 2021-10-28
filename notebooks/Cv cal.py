import glob

cv = []

for i in glob.glob(r"../input/swin-transformenrs-pet-net/*.pth"):
	i = i.split("_")
	print(i[9])
	cv.append(float(i[9]))
mean = sum(cv) / len(cv)
print(mean)
