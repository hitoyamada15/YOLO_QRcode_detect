from ctypes import *
import random
import os
import cv2  
import numpy as np
from cv2 import aruco
import time
import math
import datetime
import darknet
import argparse
from pyzbar.pyzbar import decode, ZBarSymbol
from threading import Thread, enumerate
from queue import Queue


input_path = "YOLO_QRcode_detect/USB_4K_V2_EQ_0.jpg"

CONFIG_file  = "./YOLO_QRcode_detect/yolov4_qr_code_20210903_1615.cfg"
DATA_file    = "./YOLO_QRcode_detect/obj.data"
WEIGHTS_file = "./YOLO_QRcode_detect/backup/yolov4_qr_code_20210903_1615_final.weights"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ARマーカーの番号
AR_top_left     = 0
AR_top_right    = 1
AR_bottom_right = 2
AR_bottom_left  = 3

# 台形補正 比率調整
W_ratio = 0.8

# ARマーカーから4点の座標を取得
def Aruco_detect(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	
	top_left_xy_list = []
	top_right_xy_list = []
	bottom_left_xy_list = []
	bottom_right_xy_list = []
	# ARマーカー検知
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	# 座標とidの確認
	for i in range(len(ids)):
		# 検知したidの4点取得
		c = corners[i][0]
		x1, x2, x3, x4 = c[:, 0]
		y1, y2, y3, y4 = c[:, 1]

		print(f"id={ids[i]}")
		print("X座標", x1, x2, x3, x4)
		print("Y座標", y1, y2, y3, y4)
		print("中心座標", c[:, 0].mean(), c[:, 1].mean())

		if ids[i] == AR_top_left: # 左上
			top_left_x = int(c[:, 0].mean())
			top_left_y = int(c[:, 1].mean())
			top_left_xy_list = np.append(top_left_xy_list, top_left_x)
			top_left_xy_list = np.append(top_left_xy_list, top_left_y)
		elif ids[i] == AR_top_right: # 右上
			top_right_x = int(c[:, 0].mean())
			top_right_y = int(c[:, 1].mean())
			top_right_xy_list = np.append(top_right_xy_list, top_right_x)
			top_right_xy_list = np.append(top_right_xy_list, top_right_y)
		elif ids[i] == AR_bottom_left: # 右下
			bottom_left_x = int(c[:, 0].mean())
			bottom_left_y = int(c[:, 1].mean())
			bottom_left_xy_list = np.append(bottom_left_xy_list, bottom_left_x)
			bottom_left_xy_list = np.append(bottom_left_xy_list, bottom_left_y)
		elif ids[i] == AR_bottom_right: # 左下
			bottom_right_x = int(c[:, 0].mean())
			bottom_right_y = int(c[:, 1].mean())
			bottom_right_xy_list = np.append(bottom_right_xy_list, bottom_right_x)
			bottom_right_xy_list = np.append(bottom_right_xy_list, bottom_right_y)
	
	# 検知箇所を画像にマーキング
	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	frame_markers = cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB)
	#cv2.imwrite("./aruco_detect/"+ save_time +"_aruco_detect.jpg", frame_markers)

	"""
	# もしARマーカーを読み取れないときに使う
	top_left_xy_list = np.array([781, 76])
	top_right_xy_list = np.array([2472, 766])
	bottom_left_xy_list = np.array([641, 2415])
	bottom_right_xy_list = np.array([2469, 1860])
	"""
	
	return top_left_xy_list, top_right_xy_list, bottom_left_xy_list, bottom_right_xy_list

#台形補正
def Trapezoid_correction(img, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy):
	# 変換前4点の座標 p1:左上 p2:右上 p3:左下 p4:右下
	p1 = np.array(top_left_xy)
	p2 = np.array(top_right_xy)
	p3 = np.array(bottom_left_xy)
	p4 = np.array(bottom_right_xy)
	# 幅取得
	o_width = np.linalg.norm(p2 - p1)
	o_width = math.floor(o_width * W_ratio)
	# 高さ取得
	o_height = np.linalg.norm(p3 - p1)
	o_height = math.floor(o_height)
	# 変換前の4点
	src = np.float32([p1, p2, p3, p4])
	# 変換後の4点
	dst = np.float32([[0, 0],[o_width, 0],[0, o_height],[o_width, o_height]])
	# 変換行列
	M = cv2.getPerspectiveTransform(src, dst)
	# 射影変換・透視変換する
	output = cv2.warpPerspective(img, M,(o_width, o_height))
	# 射影変換・透視変換した画像の保存
	# cv2.imwrite("./trapezoid_correction/"+ save_time + "_trapezoid_correction.jpg", output)

	return output


def convert2relative(bbox):
	"""
	YOLO format use relative coordinates for annotation
	"""
	x, y, w, h  = bbox
	_height     = darknet_height
	_width      = darknet_width
	return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
	x, y, w, h = convert2relative(bbox)

	image_h, image_w, __ = image.shape

	orig_x       = int(x * image_w)
	orig_y       = int(y * image_h)
	orig_width   = int(w * image_w)
	orig_height  = int(h * image_h)

	bbox_converted = (orig_x, orig_y, orig_width, orig_height)

	return bbox_converted


def convert4cropping(image, bbox):
	x, y, w, h = convert2relative(bbox)

	image_h, image_w, __ = image.shape

	orig_left    = int((x - w / 2.) * image_w)
	orig_right   = int((x + w / 2.) * image_w)
	orig_top     = int((y - h / 2.) * image_h)
	orig_bottom  = int((y + h / 2.) * image_h)

	if (orig_left < 0): orig_left = 0
	if (orig_right > image_w - 1): orig_right = image_w - 1
	if (orig_top < 0): orig_top = 0
	if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

	bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

	return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
								interpolation=cv2.INTER_LINEAR)
	frame_queue.put(frame)
	img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
	darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
	darknet_image_queue.put(img_for_detect)


def inference(darknet_image_queue, detections_queue, fps_queue):
	darknet_image = darknet_image_queue.get()
	detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.7)# thresh=閾値
	detections_queue.put(detections)
	darknet.free_image(darknet_image)

# YOLOの検出→QRコードの読み取り
def drawing(frame_queue, detections_queue, fps_queue):
	random.seed(3)  # deterministic bbox colors

	frame = frame_queue.get()
	detections = detections_queue.get()
	print(detections)

	detections_adjusted = []
	count = 1
	if frame is not None:
		for label, confidence, bbox in detections:
			bbox_adjusted = convert2original(frame, bbox)
			detections_adjusted.append((str(label), confidence, bbox_adjusted))

			# 検出範囲から中心座標と高さと幅を取得
			bbox_adjusted = str(bbox_adjusted)
			bbox_adjusted = bbox_adjusted.lstrip("(")
			bbox_adjusted = bbox_adjusted.rstrip(")")
			bbox_adjusted = bbox_adjusted.replace(' ', '')
			print(bbox_adjusted)
			bbox_list = bbox_adjusted.split(",")
			ceter_x = int(bbox_list[0])
			ceter_y = int(bbox_list[1])
			width   = int(bbox_list[2])
			hight   = int(bbox_list[3])

			# 切り取る座標を計算(末尾の±5は5ピクセル分大きく切り取らせるため)
			top_x    = int(ceter_x-(width/2)-5)
			top_y    = int(ceter_y-(hight/2)-5)
			bottom_x = int(ceter_x+(width/2)+5)
			bottom_y = int(ceter_y+(hight/2)+5)

			img = frame[top_y: bottom_y ,top_x : bottom_x] # YOLOで検出したQRコードを切り取り

			value = decode(img, symbols=[ZBarSymbol.QRCODE])

			if value:
				for qrcode in value:
					print("count -->",count)
					count += 1
					x, y, w, h = qrcode.rect
					# QRコードデータ
					dec_inf = qrcode.data.decode('utf-8')
					print('dec:', dec_inf)
					img_bgr = cv2.putText(img, dec_inf, (x, y - 6), FONT, .3, (255, 0, 0), 1, cv2.LINE_AA)
					# バウンディングボックス
					cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			x_offset=top_x
			y_offset=top_y
			frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

	cv2.imwrite("IZ_QR/"+input_path+"_detect.jpg",frame)


if __name__ == '__main__':
	frame_queue = Queue()
	darknet_image_queue = Queue(maxsize=1)
	detections_queue = Queue(maxsize=1)
	fps_queue = Queue(maxsize=1)

	network, class_names, class_colors = darknet.load_network(
		CONFIG_file,
		DATA_file,
		WEIGHTS_file,
		batch_size=1
		)
	darknet_width = darknet.network_width(network)
	darknet_height = darknet.network_height(network)

	input_img = cv2.imread(input_path)
	# 関数(Aruco_detect)
	top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy = Aruco_detect(input_img)
	# 関数(Trapezoid_correction)				
	output_img = Trapezoid_correction(input_img, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)

	frame = output_img

	Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
	Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
	Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()

