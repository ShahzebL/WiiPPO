import numpy as np
import matplotlib
from PIL import ImageGrab
import cv2
from collections import deque

BBOX = (0, 60, 1280, 1020)
# img = ImageGrab.grab(
#     bbox=BBOX, 
#     include_layered_windows=False, 
#     all_screens=False, 
#     xdisplay=None
# )
# img.show()
observations = deque(maxlen = 4)
while True:
	img = ImageGrab.grab(
	    bbox=BBOX, 
	    include_layered_windows=False, 
	    all_screens=False, 
	    xdisplay=None
	)
	img = np.array(img)
	scale_percent = 10 # percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	  
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	observations.append(resized)
	frames = np.stack(observations)
	print(frames.shape)
	# img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	cv2.imshow('recording', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
	cv2.waitKey(1)
