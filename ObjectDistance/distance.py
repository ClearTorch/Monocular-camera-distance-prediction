'''
Create July 27, 2020
@author ClearTorch
'''

import pandas as pd
import numpy as np
import cv2
import sys
import math
import os

import plotly.graph_objects as go

print('Pandas Version:', pd.__version__)
print('Nunpy Version:', np.__version__)

class DistanceEstimation:
	'''
	DistanceEstimation类用于对检测到的目标进行距离预测
	'''
	def __init__(self):
		'''
		初始化图片分辨率及相机标定数据
		'''
		self.W = 1920
		self.H = 1208
		self.excel_path = r'./camera_parameters.xlsx'
		self.txt_path = r'./inference/output/'

	def camera_parameters(self, excel_path):
		'''
		camera_parameters()函数对标定的相机参数Excel表进行读取及处理
		:param excel_path: Excel表的绝对路径(相机参数标定Excel由Matlab得到)
		:return: 返回相机标定的外参矩阵,和内参矩阵
		'''
		#df_rotation = pd.read_excel(excel_path, sheet_name = '旋转矩阵', header=None)
		#df_trans = pd.read_excel(excel_path, sheet_name = '平移矩阵', header=None)
		df_intrinsic = pd.read_excel(excel_path, sheet_name = '内参矩阵', header=None)
		df_p =  pd.read_excel(excel_path, sheet_name = '外参矩阵', header=None)

		print('外参矩阵形状',df_p.values.shape)
		print('内参矩阵形状：',df_intrinsic.values.shape)

		return df_p.values, df_intrinsic.values


	def scale_factor(self,u,v, w, h, p, k, xw, yw, zw=0):
		'''
		scale_factor()函数用于求解关于目标测距关键点C的相关性未知尺度因子S
		目标检测结果得到待测目标的矩形框位置(u v w h)，其中(u v)表示矩形框在图像中左上角顶点的坐标值，(w h)表示矩形框的宽度和高度像素值；
		测距关键点C的世界坐标为(Xw，Yw，Zw)，由于所计算的关键点C的世界坐标点位于水平地面上，所以Zw＝0
		:param u: 检测目标矩形框的左上角x坐标像素值
		:param v: 检测目标矩形框的左上角y纵标像素值
		:param w: 检测目标矩形框的高度像素值
		:param h: 检测目标矩形框的高度像素值
		:param p: 外参矩阵
		:param k: 内参矩阵
		:param xw: 目标测距关键点的物理x坐标
		:param yw: 目标测距关键点的物理y坐标
		:param zw: 目标测距关键点的物理z坐标
		:return: 目标测距关键点C的相关性未知尺度因子S
		'''
		u1 = u+w/2
		v1 = v+h/2
		if 0 < u1 and u1 < self.W and 0 < v1 and v1 < self.H:
			print('='*50)
		else:
			print('目标测距关键点C坐标不合法')
			exit()
		world_mat = np.array([xw,yw,zw,1])
		world_mat = np.transpose(world_mat)
		point_c = np.transpose(np.array([u1,v1,1,1]))
		right = np.matmul(np.matmul(k,p),world_mat)

		s = (right/point_c)
		return s

	def object_point_world_position(self, s, u,v, w, h, p, k):
		'''
		object_point_world_position()函数根据物体的检测检测框及相机未知尺度因子等参数求出测距关键点的世界坐标
		:param s: 目标测距关键点C的相关性未知尺度因子S
		:param u: 检测目标矩形框的左上角x坐标像素值
		:param v: 检测目标矩形框的左上角y纵标像素值
		:param w: 检测目标矩形框的高度像素值
		:param h: 检测目标矩形框的高度像素值
		:param p: 外参矩阵
		:param k: 内参矩阵
		:return: 测距关键点的世界坐标
		'''
		u1 = u+w/2
		v1 = v+h/2
		point_c = np.array([u1, v1, 1, 1])
		point_c = np.transpose(point_c)
		kp_inv = np.linalg.inv(np.matmul(k, p))
		c_position = np.matmul(kp_inv, s * point_c)
		d1 = (c_position[1], c_position[2])
		return d1

	def center_point_world_position(self, u,  w,  xw, k):
		'''
		object_size()函数根据相似三角形原理求出需要求的目标纵向和横向物理距离
		:param u: 检测目标矩形框的左上角x坐标像素值
		:param w: 检测目标矩形框的高度像素值
		:param ww: 为目标实际物理宽度
		:param k: 内参矩阵
		:param dx: 为相机横向像素单位大小
		:return: 需要求的目标纵向和横向物理距离，记为D2(X2，Y2)
		'''

		l =  np.abs(u - self.W / 2)
		fx = k[0,0]

		r = w / xw
		x2 = fx / r
		y2 = l / r
		d2 = (x2, y2)

		return d2

	def error_correction(self, d1, d2):
		'''
		对距离进行卡尔曼滤波修正
		:param d1: 测距关键点世界坐标
		:param d2:  目标中心点世界坐标
		:return: 修正后的目标世界坐标
		'''
		#print('d1=',d1, 'd2=',d2)
		x = 0.3 * d1[0] + d2[0] * 0.1
		y = d1[1]*0.9 + 0.1*d2[1]
		distance = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
		distance = round(distance, 3)
		x = round(x, 3)
		y = round(y, 3)
		return distance, x, y

	def detect_info(self, txt_path):
		'''
		detect_info()函数提取猫狗的检测结果
		:param txt_path: 检测结果路径（也可以为模型检测函数接口，但要考虑协调性）
		:return: 返回猫狗的检测信息
		'''
		dog = {}
		cat = {}
		file_names= os.listdir(txt_path)
		for name in file_names:
			if name[-4:] == '.txt':
				num = list(filter(str.isdigit, name))
				num = "".join(num)
				with open(txt_path + name, 'r') as txt:
					info = txt.readlines()
					for stri in info:
						if stri[0] == '0':
							dog[eval(num)] = stri.strip('\n')
						elif stri[0] == '1':
							cat[eval(num)] = stri.strip('\n')
						else:
							pass
				os.remove(txt_path+name)
		return dog, cat

	def distance(self, xw = 5, yw = 0.1):

		print('开始猫狗测距')
		print('=' * 50)
		fig = go.Figure()
		x_dog = []
		y_dog = []
		x_cat = []
		y_cat = []
		p, k = self.camera_parameters(self.excel_path)
		dogs, cats = self.detect_info(self.txt_path)
		if len(dogs) != 0:
			dog_position = {}
			keys = list(dogs.keys())
			keys.sort()
			dog = dogs[keys[0]].split(' ')
			u, v, w, h = eval(dog[1]) * self.W, eval(dog[2]) * self.H, eval(dog[3]) * self.W, eval(dog[4]) * self.H
			s_dog = self.scale_factor(u, v, w, h, p, k, xw, yw)
			dog_position[0] = (xw, yw)
			for key in keys[1:]:
				dog = dogs[key]
				dog = dog.split(' ')
				u, v, w, h = eval(dog[1]) * self.W, eval(dog[2]) * self.H, eval(dog[3]) * self.W, eval(dog[4])* self.H
				d1 = self.object_point_world_position(s_dog, u, v, w, h, p, k)
				d2 = self.center_point_world_position( u,  w,  xw, k)
				distance, x, y = self.error_correction(d1, d2)
				print('图片名：',str(key)+'.png')
				print('距离{0}直线距离{1} m，物体相对坐标为{2}'.format('狗', distance,(x, y)))
				print('=='*30)
				x_dog.append(x)
				y_dog.append(y)
			fig.add_trace(go.Scatter(x = y_dog, y = x_dog, mode = 'markers', name = 'Real Position',
			                         marker = dict(size = 50, color = '#FF3030')))

		else:
			print('图像中没有狗！')

		if len(cats) != 0:
			cat_position = {}
			keys = list(cats.keys())
			keys.sort()
			cat = cats[keys[0]].split(' ')
			u, v, w, h = eval(cat[1]) * self.W, eval(cat[2]) * self.H, eval(cat[3]) * self.W, eval(cat[4]) * self.H
			s_cat = self.scale_factor(u, v, w, h, p, k, xw, yw)
			cat_position[0] = (xw, yw)
			for key in keys[1:]:
				cat = cats[key]
				cat = cat.split(' ')
				u, v, w, h = eval(cat[1]) * self.W, eval(cat[2]) * self.H, eval(cat[3]) * self.W, eval(cat[4]) * self.H

				d1 = self.object_point_world_position(s_cat, u, v, w, h, p, k)
				d2 = self.center_point_world_position(u, w, xw, k)
				distance, x, y = self.error_correction(d1, d2)
				print('图片名：', str(key) + '.png')
				print('距离{0}直线距离{1} m，物体相对坐标为{2}'.format('猫', distance, (x, y)))
				print('==' * 30)
				x_cat.append(x)
				y_cat.append(y)

			fig.add_trace(go.Scatter(x = y_cat, y = x_cat, mode = 'markers', name = 'Real Position',
			                         marker = dict(size = 20, color = '#FF7F24')))
		else:
			print('图像中没有猫')
		fig.update_layout(title = '猫狗距离预测图',xaxis_title = '左右偏移量/m', yaxis_title = '垂直距离/m', font =dict(size = 60))
		fig.show()

		if os.path.exists(self.txt_path+'position.txt'):
			os.remove(self.txt_path+'position.txt')
		else:
			with open(self.txt_path+'position.txt', 'w') as pf:
				if len(x_dog) !=0:
					pf.write(' '*10 + '狗的位置'+ '\n')
					pf.write('x坐标： '+ str(x_dog))
					pf.write('\n')
					pf.write('y坐标： ' + str(y_dog))
					pf.write('\n')
				if len(x_cat) != 0:
					pf.write(' '*10 + '猫的位置'+ '\n')
					pf.write('x坐标： '+ str(x_cat))
					pf.write('\n')
					pf.write('y坐标： ' + str(y_cat))
					pf.write('\n')

		return

	def video2picture(self, video_src_path,  frame_save_path):
		'''
		把视频转换问图像帧
		:param video_src_path: 视频文件路径
		:param frame_save_path: 图像保存路径
		:return:
		'''
		vc = cv2.VideoCapture(video_src_path)
		c = 1
		if vc.isOpened():
			rval, frame = vc.read()
		else:
			rval = False
		while rval:
			rval, frame = vc.read()
			cv2.imwrite(frame_save_path + str(c) + '.jpg', frame)
			c = c + 1
			cv2.waitKey(1)
		vc.release()

class ObjectTracking:
	'''
	基于Opencv的传统目标追踪
	'''
	def __init__(self):
		pass

	def single_target_tracking(self, object_path):
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		print(major_ver, minor_ver, subminor_ver)

		# 创建跟踪器
		tracker_type = 'MIL'
		tracker = cv2.TrackerMIL_create()
		# 读入视频
		#camera = cv2.VideoCapture(0)
		#video = camera
		video = cv2.VideoCapture(object_path)

		# 读入第一帧
		ok, frame = video.read()
		if not ok:
			print('Cannot read video file')
			sys.exit()
		# 定义一个bounding box
		bbox = (287, 23, 86, 320)
		bbox = cv2.selectROI(frame, False)
		# 用第一帧初始化
		ok = tracker.init(frame, bbox)

		while True:
			ok, frame = video.read()
			if not ok:
				break
			# Start timer
			timer = cv2.getTickCount()
			# Update tracker
			ok, bbox = tracker.update(frame)
			# Cakculate FPS
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			# Draw bonding box
			if ok:
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
			else:
				cv2.putText(frame, "Tracking failed detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
				            2)
			# 展示tracker类型
			cv2.putText(frame, tracker_type + "Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
			# 展示FPS
			cv2.putText(frame, "FPS:" + str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
			# Result
			cv2.imshow("Tracking", frame)

			# Exit
			k = cv2.waitKey(1) & 0xff
			if k == 27: break

	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	print(major_ver, minor_ver, subminor_ver)

	def multi_target_tracking(self, object_path):
		# 创建跟踪器
		# 'BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE'
		tracker_type = 'MIL'
		tracker = cv2.MultiTracker_create()
		# 创建窗口
		cv2.namedWindow("Tracking")
		# 读入视频
		video = cv2.VideoCapture(object_path)
		# 读入第一帧
		ok, frame = video.read()
		if not ok:
			print('Cannot read video file')
			sys.exit()
		# 定义一个bounding box
		box1 = cv2.selectROI("Tracking", frame)
		box2 = cv2.selectROI("Tracking", frame)
		box3 = cv2.selectROI("Tracking", frame)
		# 用第一帧初始化
		ok = tracker.add(cv2.TrackerMIL_create(), frame, box1)
		ok1 = tracker.add(cv2.TrackerMIL_create(), frame, box2)
		ok2 = tracker.add(cv2.TrackerMIL_create(), frame, box3)
		while True:
			ok, frame = video.read()
			if not ok:
				break
			# Start timer
			timer = cv2.getTickCount()
			# Update tracker
			ok, boxes = tracker.update(frame)
			print(ok, boxes)
			# Cakculate FPS
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			for box in boxes:
				# Draw bonding box
				if ok:
					p1 = (int(box[0]), int(box[1]))
					p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
					cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
				else:
					cv2.putText(frame, "Tracking failed detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
					            (0, 0, 255),
					            2)
			# 展示tracker类型
			cv2.putText(frame, tracker_type + "Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
			# 展示FPS
			cv2.putText(frame, "FPS:" + str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
			# Result
			cv2.imshow("Tracking", frame)

			# Exit
			k = cv2.waitKey(1) & 0xff
			if k == 27: break

if __name__ == '__main__':
	print('开始进行目标检测')
	print('查看检测结果（数据量少， 模型欠拟合）')
	os.system('python detect.py --view-img')

	print('猫狗检测完成，开始进行深度估计')
	os.system('python test_simple.py --image_path assets/1.png --model_name mono_640x192')

	print('深度估计完成， 开始进行基于坐标变换的距离预测')
	DE = DistanceEstimation()
	print('猫狗距离预测结果！')
	DE.distance()

	img1 = cv2.imread('assets/1.png')
	img2 = cv2.imread('assets/1_disp.jpeg')
	imgs = np.hstack([img1, img2])
	cv2.namedWindow("Monocular Depth Estimation", 1000)
	cv2.imshow("Monocular Depth Estimation", imgs)
	img3 = cv2.imread('result.png')

	cv2.namedWindow("Distance Estimation Result", 2000)
	cv2.imshow("Distance Estimation Result", img3)
	cv2.waitKey (0)
	cv2.destroyAllWindows()

	# OT = ObjectTracking()
	# object_path = 'D://PythonFile/shangqi/yolov5/inference/images/runman.mp4'
	# OT.single_target_tracking(object_path)
	# OT.multi_target_tracking(object_path)








