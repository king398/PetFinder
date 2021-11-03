from tqdm import tqdm
z = 0
num = range(11111111111111111, 11111111119111111)

for p in tqdm(num):
	num = p
	num_save = num
	reversed_num = 0

	while num != 0:
		digit = num % 10
		reversed_num = reversed_num * 10 + digit
		num //= 10

	dim = num_save + reversed_num
	dim = list(map(int, str(dim)))
	for i in dim:
		if i % 2 == 0:
			z += 1
			break
num = range(11111111111111111, 11111111111911111)

print("done")
if len(num) == z:
	print(z)