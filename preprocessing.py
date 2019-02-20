# image preprocessing, run in folder to convert all images to grayscale

import numpy as np
import cv2
import os

dir_path = os.getcwd()


kernel = np.matrix('0 1 0; 1 1 1; 0 1 0')
kernel = np.ones_like(kernel,np.uint8)
kernel[0,0] = 0
kernel[0,2] = 0
kernel[2,0] = 0
kernel[2,2] = 0
#print(kernel)

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPEG") or filename.endswith(".png") or filename.endswith(".PNG"):
        image = cv2.imread(filename)
        converted = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        converted = cv2.adaptiveThreshold(converted,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #kernel = np.ones((3,3),np.uint8)
        converted = cv2.bitwise_not(converted)
        converted = cv2.erode(converted,kernel,iterations = 1)
        #print(kernel)
        #converted = cv2.dilate(converted,kernel,iterations = 1)
        #converted = cv2.erode(converted,kernel,iterations = 1)
        cv2.imwrite(filename,converted)
