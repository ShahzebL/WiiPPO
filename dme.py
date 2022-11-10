import dolphin_memory_engine as dme
import time

dme.hook()
print(dme.is_hooked())
i = 0
time.sleep(7)
while True:
	#136 192
	num = dme.read_float(2162427364)
	# print(i)
	print(num)
	# time.sleep(1)
	# for j in range(10):
	# 	num = dme.read_float(2163835920 + i)
	# 	print(num)
	# 	time.sleep(0.1)
	# time.sleep(0.5)

	# if num >= 84.00999 and num <= 84.01:
	# 	for j in range(10):
	# 		num = dme.read_float(2162464076 + i)
	# 		print(num)
	# 		print(i)
	# 		time.sleep(1)
	if i % 251658 == 0:
		print(i)
	i += 4