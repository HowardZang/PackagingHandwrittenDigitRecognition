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
import RegionGrow as rg


def separate_numbers(img):
	block_size = 160
	rows, cols = img.shape
	minimum = np.min(img)
	seeds = []
	binaryImg = []
	k = 0
	# 持续迭代，直到roi中不存在黑像素
	while (minimum == 0):
		# 遍历roi寻找种子像素
		for i in range(rows):
			for j in range(cols):
				if img[i, j] == 0:
					seeds = [rg.Point(i, j)]
					break
		# 若找到种子像素
		if len(seeds) > 0:
			binaryImg.append(rg.regionGrow(img, seeds, 10))
		else:
			print('未找到种子像素')
			break
		# 将分离出来的数字，从roi中涂白抹除
		for i in range(rows):
			for j in range(cols):
				if binaryImg[k][i, j] != 0.0:
					img[i, j] = 255
		# 再次迭代
		minimum = np.min(img)
		k = k + 1
	binaryImg255 = np.zeros(np.array(binaryImg).shape, dtype=np.uint8)
	for i in range(len(binaryImg)):
		for j in range(rows):
			for k in range(cols):
				if binaryImg[i][j, k] == 0.0:
					binaryImg255[i][j, k] = 255
				else:
					binaryImg255[i][j, k] = 0

	# 数字框选
	nums = []
	pos = []
	for k in range(len(binaryImg255)):
		# 纵向查找
		black_flag = 0
		min_i = 0
		min_j = 0
		max_i = rows
		max_j = cols
		for j in range(cols):
			temp = binaryImg255[k][:, j]
			# 如果j列之前均为白色像素
			if black_flag == 0:
				# j列存在黑色像素，black_flag置1，同时记录此时列数j为min_j
				if np.min(temp) == 0:
					black_flag = 1
					min_j = j
				# j列不存在黑色像素，则继续遍历
			# 如果j列之前存在黑色像素
			elif black_flag == 1:
				# j列不存在黑色像素，black_flag置0，同时记录此时列数j为max_j
				if np.min(temp) == 255:
					black_flag = 0
					max_j = j
			# j列仍然存在黑色像素，则继续遍历
		# 横向查找
		black_flag = 0
		for i in range(rows):
			temp = binaryImg255[k][i, :]
			# 如果i行之前均为白色像素
			if black_flag == 0:
				# i行存在黑色像素，black_flag置1，同时记录此时行数i为min_i
				if np.min(temp) == 0:
					black_flag = 1
					min_i = i
			# i行不存在黑色像素，则继续遍历
			# 如果i行之前存在黑色像素
			elif black_flag == 1:
				# i行不存在黑色像素，black_flag置0，同时记录此时列数i为max_i
				if np.min(temp) == 255:
					black_flag = 0
					max_i = i
		# i行仍然存在黑色像素，则继续遍历
		pic = binaryImg255[k][min_i:max_i, min_j:max_j]
		pic = cv2.resize(pic, (block_size, block_size), interpolation=cv2.INTER_AREA)
		nums.append(pic)
		pos.append([min_i, min_j, max_i, max_j])
	nums = np.array(nums)
	return nums, pos


def fill_blanks(nums):
	nums = np.array(nums)
	block_size = 280
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


def main():
	# 读取图片
	img = cv2.imread('pictures/box3.jpg')
	# 将尺寸变换到合适的大小
	rows = 960
	cols = 540
	img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_AREA)
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
	[min_i, min_j, max_i, max_j] = rg.select_corner(dst)
	cv2.rectangle(img, (min_j-1, min_i-1), (max_j-1, max_i-1), (0, 255, 0), 2)  # 以绿色在原图img上绘制ROI区域
	cv2.putText(img, 'Express package', (min_j, min_i - 5), cv2.FONT_ITALIC, 0.75, (0, 255, 0), 2)
	# 根据角点提取出感兴趣区域ROI
	roi = binary[min_i:max_i, min_j:max_j]
	roi = np.uint8(roi)
	# 对ROI进行闭运算，以消除边缘噪声
	kernel = np.ones((5, 5), np.uint8)
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
	# 从ROI中分离出各个数字，以list形式存储
	nums, pos = separate_numbers(roi)
	for i in range(len(nums)):
		cv2.imshow('nums[%d]' % i, nums[i])
	# for i in range(len(nums)):
	# 	cv2.imshow('nums[%d]' % i, nums[i]  )
	# 在分离出的各个数字周围以白色像素填充
	nums = fill_blanks(nums)
	for i in range(len(nums)):
		cv2.imshow('nums[%d]' % i, nums[i])
	# for i in range(len(nums)):
	# 	cv2.imshow('blank_nums[%d]' % i, nums[i])
	# 计算并显示结果
	result = []
	final = 0
	length = len(nums)
	for i in range(length):
		# 计算过程
		number = mnist_lenet5_app.application(nums[i])
		result.append(number)
		final = final + int(result[i]) * pow(10, (length - 1) - i)
		# 绘制图像
		cv2.rectangle(img, (min_j+pos[i][1]-2, min_i+pos[i][0]-2), (min_j+pos[i][3]-2, min_i+pos[i][2]-2), (0, 0, 255), 2)
		cv2.putText(img, 'nums[%d]=%d' % (i, number), (min_j+pos[i][1], min_i+pos[i][0] - 7), cv2.FONT_ITALIC, 0.75, (0, 0, 255), 2)
	cv2.putText(img, 'The result is %d' % final, (0, cols-2), cv2.FONT_ITALIC, 0.75, (255, 255, 255), 2)

	cv2.imshow('img', img)
	# cv2.imshow('binary', binary)
	# cv2.imshow('dst', dst)
	# cv2.imshow('roi', roi)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
