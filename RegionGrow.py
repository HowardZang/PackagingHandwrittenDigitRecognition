#!D:\Software\Anaconda\InstallAddress\envs\tensorflow\python.exe
# -*- coding: utf-8 -*- 
# @Time : 2019/5/11 12:48 
# @Author : Howard 
# @Site :  
# @File : RegionGrow.py 
# @Software: PyCharm
import numpy as np
import cv2


class Point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def getX(self):
		return self.x

	def getY(self):
		return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
	return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
	if p != 0:
		connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
		            Point(0, 1), Point(-1, 1), Point(-1, 0)]
	else:
		connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
	return connects


def regionGrow(img, seeds, thresh, p=1):
	height, weight = img.shape
	seedMark = np.zeros(img.shape)
	seedList = []
	for seed in seeds:
		seedList.append(seed)
	label = 1
	connects = selectConnects(p)
	while (len(seedList) > 0):
		currentPoint = seedList.pop(0)

		seedMark[currentPoint.x, currentPoint.y] = label
		for i in range(8):
			tmpX = currentPoint.x + connects[i].x
			tmpY = currentPoint.y + connects[i].y
			if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
				continue
			grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
			if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
				seedMark[tmpX, tmpY] = label
				seedList.append(Point(tmpX, tmpY))
	return seedMark


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
	[min_i, min_j, max_i, max_j] = select_corner(dst)
	# 根据角点提取出感兴趣区域ROI
	roi = binary[min_i:max_i, min_j:max_j]
	roi = np.uint8(roi)
	# 对ROI进行闭运算，以消除边缘噪声
	kernel = np.ones((5, 5), np.uint8)
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
	rows, cols = roi.shape
	minimum = np.min(roi)
	seeds = []
	binaryImg = []
	k = 0
	# 持续迭代，直到roi中不存在黑像素
	while (minimum == 0):
		# 遍历roi寻找种子像素
		for i in range(rows):
			for j in range(cols):
				if roi[i, j] == 0:
					seeds = [Point(i, j)]
					break
		# 若找到种子像素
		if len(seeds) > 0:
			binaryImg.append(regionGrow(roi, seeds, 10))
		else:
			print('未找到种子像素')
			break
		# 将分离出来的数字，从roi中涂白抹除
		for i in range(rows):
			for j in range(cols):
				if binaryImg[k][i, j] != 0.0:
					roi[i, j] = 255
		# 再次迭代
		minimum = np.min(roi)
		k = k + 1
	binaryImg255 = np.zeros(np.array(binaryImg).shape, dtype=np.uint8)
	for i in range(len(binaryImg)):
		for j in range(rows):
			for k in range(cols):
				if binaryImg[i][j, k] == 0.0:
					binaryImg255[i][j, k] = 255
				else:
					binaryImg255[i][j, k] = 0

	cv2.imshow('binaryImg255[0]', binaryImg255[0])
	cv2.imshow('binaryImg255[1]', binaryImg255[1])
	cv2.imshow('roi', roi)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()
