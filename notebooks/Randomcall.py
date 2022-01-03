import random


def first():
	return 1


def second():
	return 2


def third():
	return 3


func_dict = {"first": first(), "second": second(), "third": third()}
ls = ["first", "second", "third"]
choice = random.choices(ls, k=1)
for i in choice:
	func = func_dict[i]
	result = func
	print(result)