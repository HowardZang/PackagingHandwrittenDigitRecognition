#!D:\Software\Anaconda\InstallAddress\envs\tensorflow\python.exe
# -*- coding: utf-8 -*- 
# @Time : 2019/4/14 16:53 
# @Author : Howard 
# @Site :  
# @File : Image_Preprocessing.py 
# @Software: PyCharm
import cv2
import numpy as np
import mnist_lenet5_app
# 读取图片
img_origin = cv2.imread('pictures/box3.jpg')
# 将尺寸变换到合适的大小
rows = 960
cols = 540
img = cv2.resize(img_origin, (rows, cols), interpolation=cv2.INTER_AREA)


def select_corner(img):
	[rows, cols] = img.shape  # [行， 列]
	record_min_i = rows
	record_min_j = cols
	record_max_i = 0
	record_max_j = 0
	threshold = 0.01*img.max()
	# 在第一个象限中寻找最接近左上端的角点
	minimum = rows + cols
	record_i1 = rows
	record_j1 = cols
	for i in range(int(rows/2)):
		for j in range(int(cols/2)):
			if img[i, j] > threshold and i + j < minimum:
				minimum = i + j
				record_i1 = i
				record_j1 = j
	# print('第一象限角点为', record_i1, record_j1)
	# 在第二个象限中寻找最接近右上端的角点
	maximum = 0
	record_i2 = 0
	record_j2 = cols
	for i in range(int(rows/2)):
		for j in range(cols - 1, int(cols/2), -1):
			if img[i, j] > threshold and j / (i+1) > maximum:
				maximum = j / (i+1)
				record_i2 = i
				record_j2 = j
	# print('第二象限角点为', record_i2, record_j2)
	# 在第三个象限中寻找最接近左下端的角点
	maximum = 0
	record_i3 = rows
	record_j3 = 0
	for i in range(rows - 1, int(rows/2), -1):
		for j in range(int(cols/2)):
			if img[i, j] > threshold and i / (j+1) > maximum:
				maximum = i / (j+1)
				record_i3 = i
				record_j3 = j
	# print('第三象限角点为', record_i3, record_j3)
	# 在第四个象限中寻找最接近右下端的角点
	maximum = 0
	record_i4 = 0
	record_j4 = 0
	for i in range(rows - 1, int(rows/2), -1):
		for j in range(cols - 1, int(cols/2), -1):
			if img[i, j] > threshold and i + j > maximum:
				maximum = i + j
				record_i4 = i
				record_j4 = j
	# print('第四象限角点为', record_i4, record_j4)
	# 判定最佳ROI范围
	record_min_i = max(record_i1, record_i2)
	record_min_j = max(record_j1, record_j3)
	record_max_i = min(record_i3, record_i4)
	record_max_j = min(record_j2, record_j4)
	# print('The return value is ', record_min_i, record_min_j, record_max_i, record_max_j)
	return [record_min_i, record_min_j, record_max_i, record_max_j]


def separate_numbers(img):
	block_size = 16
	rows, cols = img.shape
	min_i = 0
	min_j = []
	max_i = rows
	max_j = []
	# 纵向查找
	black_flag = 0
	for j in range(cols):
		temp = img[:, j]
		# 如果j列之前均为白色像素
		if black_flag == 0:
			# j列存在黑色像素，black_flag置1，同时记录此时列数j为min_j
			if np.min(temp) == 0:
				black_flag = 1
				min_j.append(j)
				# print('    记录min_j位置：', j)
			# j列不存在黑色像素，则继续遍历
		# 如果j列之前存在黑色像素
		elif black_flag == 1:
			# j列不存在黑色像素，black_flag置0，同时记录此时列数j为max_j
			if np.min(temp) == 255:
				black_flag = 0
				max_j.append(j)
				# print('    记录max_j位置：', j)
			# j列仍然存在黑色像素，则继续遍历
	# 横向查找
	black_flag = 0
	for i in range(rows):
		temp = img[i, :]
		# 如果i行之前均为白色像素
		if black_flag == 0:
			# i行存在黑色像素，black_flag置1，同时记录此时行数i为min_i
			if np.min(temp) == 0:
				black_flag = 1
				min_i = i
				# print('    记录min_i位置：', i)
			# i行不存在黑色像素，则继续遍历
		# 如果i行之前存在黑色像素
		elif black_flag == 1:
			# i行不存在黑色像素，black_flag置0，同时记录此时列数i为max_i
			if np.min(temp) == 255:
				black_flag = 0
				max_i = i
				# print('    记录max_i位置：', i)
			# i行仍然存在黑色像素，则继续遍历
	print('ROI图片中存在', len(min_j), '个数字')
	# 将分离出来的数字resize成神经网络识别的28 * 28像素，并添加至返回值nums[]的尾端
	nums = []
	for i in range(len(min_j)):
		pic = img[min_i:max_i, min_j[i]:max_j[i]]
		pic = cv2.resize(pic, (block_size, block_size), interpolation=cv2.INTER_AREA)
		nums.append(pic)
	np.array(nums)
	return nums


def fill_blanks(nums):
	nums = np.array(nums)
	block_size = 28
	rows, cols = nums[0].shape
	edge = int((block_size - rows) / 2)
	imgs = []
	for i in range(len(nums)):
		# cv2.imshow('nums', nums[i])
		blank = 255 * np.ones((block_size, block_size), dtype=np.uint8)
		for j in range(rows):
			for k in range(cols):
				blank[j+edge, k+edge] = nums[i, j, k]
		# cv2.imshow('blanks', blank)
		imgs.append(blank)
	imgs = np.array(imgs)
	return imgs


# 高斯模糊
gau = cv2.GaussianBlur(img, (3, 3), 1.5)
# 根据BGR色彩二值化
binary = cv2.inRange(gau, (65, 100, 100), (230, 235, 235))
# 开运算
kernel = np.ones((12, 12), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = np.float32(binary)
# 角点检测
dst = cv2.cornerHarris(binary, 8, 3, 0.05)
# 调用select_corner()挑选出角点
[min_i, min_j, max_i, max_j] = select_corner(dst)
cv2.rectangle(img, (min_j, min_i), (max_j, max_i), (0, 255, 0), 1)  # 以绿色在原图img上绘制ROI区域
cv2.putText(img, 'Express package', (min_j, min_i-5), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
# 根据角点提取出感兴趣区域ROI
roi = binary[min_i:max_i, min_j:max_j]
roi = np.uint8(roi)
# 对ROI进行闭运算，以消除边缘噪声
kernel = np.ones((5, 5), np.uint8)
roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
# 从ROI中分离出各个数字，以list形式存储
nums = separate_numbers(roi)
# 在分离出的各个数字周围以白色像素填充
nums = fill_blanks(nums)
# 计算并显示结果
result = []
final = 0
length = len(nums)
for i in range(length):
	result.append(mnist_lenet5_app.application(nums[i]))
	final = final + int(result[i]) * pow(10, (length - 1) - i)
print('预测值为', final)

cv2.imshow('img', img)
# cv2.imshow('binary', binary)
# cv2.imshow('dst', dst)
# cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
