# import the opencv library 
import cv2 
import numpy as np


# define a video capture object 
vid = cv2.VideoCapture(0) 

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read()

	# mask = np.zeros(frame.shape[:2], np.uint8)

	# bgdModel = np.zeros((1,65), np.float64)
	# fgdModel = np.zeros((1,65), np.float64)

	# rect = (50,50,500,500)
	# cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

	# mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')
	# frame = frame * mask2[:,:,np.newaxis]
	# blurred = cv2.GaussianBlur(frame, (5,5), 0)
	# marker = np.zeros_like(img[:,:,0]).astype(np.int32)

	# imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	# edged = cv2.Canny(imgray, 30, 200)

	# contours, hierarchy = cv2.findContours(edged, 
	# 	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# frame = cv2.circle(frame, (50,50), 2, (255,255,255), -1)

	# Display the resulting frame 
	# cv2.imshow('frame', edged)
	# cv2.drawContours(frame, contours, -1, (0,255,0), 3)
	cv2.imshow('contours', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
