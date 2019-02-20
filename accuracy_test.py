######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from lxml import objectify

def convert(name):
    if(name == "aku busilica"):
        return 1
    if(name == "elektro busilica"):
        return 2
    if(name == "aku busilica"):
        return 3
    if(name == "elektro brusilica"):
        return 4
    if(name == "aku ubodna testera"):
        return 5
    if(name == "elektro ubodna testera"):
        return 6

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'elektro busilica aeg 2.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

accumulated_accuracy = 0;
number_of_tests = 0;

found = 0
not_found = 0
wrong_classification = 0

dir_path = os.getcwd() + "/images/test"
for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".JPEG") or filename.endswith(".jpeg") or filename.endswith(".PNG") or filename.endswith(".png"):
        myfile = None
        if filename.endswith(".jpeg") or filename.endswith(".JPEG"):
            myfile = open(dir_path + "/" + filename[0:len(filename)-3] + ".xml", 'r')
        else:
            myfile = open(dir_path + "/" + filename[0:len(filename)-4] + ".xml", 'r')
        data=myfile.read()
        xml_object = objectify.fromstring(data)
        image = cv2.imread(dir_path + "/" + filename)
        hight, width, channels = image.shape
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        no_of_found = 0
        for i in range(int(num)):
            if(scores[0][i] < 0.80):
                break
            no_of_found = no_of_found + 1
        partial_accuracy_correct_numbers = 1 - (abs(len(xml_object.object) - no_of_found) / len(xml_object.object))
        number_of_tests = number_of_tests + 1
        accumulated_accuracy = accumulated_accuracy + partial_accuracy_correct_numbers

        coordinates_of_found = np.array(boxes[0][0:no_of_found]) * np.array([hight, width, hight, width])
        actual_coordinates = []
        for i in range(len(xml_object.object)):
            actual_coordinates.append(np.array([int(xml_object.object[i].bndbox.ymin), int(xml_object.object[i].bndbox.xmin), int(xml_object.object[i].bndbox.ymax), int(xml_object.object[i].bndbox.xmax)]))

        # ymin xmin ymax xmax

        no_of_actual_coordinates = len(actual_coordinates)

        for i in range(no_of_actual_coordinates):
            index = 0
            overlap_pocentage = 0
            my_area = (xml_object.object[i].bndbox.xmax - xml_object.object[i].bndbox.xmin)*(xml_object.object[i].bndbox.ymax - xml_object.object[i].bndbox.ymin)
            for j in range(no_of_found):
                overlap_area = max(0, min(xml_object.object[i].bndbox.xmax, coordinates_of_found[j][3]) - max(xml_object.object[i].bndbox.xmin, coordinates_of_found[j][1])) * max(0, min(xml_object.object[i].bndbox.ymax, coordinates_of_found[i][2]) - max(xml_object.object[i].bndbox.ymin, coordinates_of_found[i][0]))
                #overlap_area = max(0, min(253, 267.8) - max(1, 3.7)) * max(0, min(623, 617) - max(255, 253.5))
                overlap_pocentage_new = overlap_area / my_area
                if overlap_pocentage_new > overlap_pocentage:
                    index = j
                    overlap_pocentage = overlap_pocentage_new
            if(overlap_pocentage > 0.75):
                found = found + 1
                if not(convert(xml_object.object[i].name) == scores[0][j]):
                    wrong_classification = wrong_classification + 1  
            else:
                not_found = not_found + 1
        

total_averaged_accuracy = accumulated_accuracy / number_of_tests

print("Counting accuracy: " + str(total_averaged_accuracy * 100) + "%")

'''print("found: " + str(found))
print("not found: " + str(not_found))
print("wrong_classification: " + str(wrong_classification))'''
found_accuracy = float(found)/(found + not_found)
print("Found accuracy: " + str(found_accuracy * 100) + "%")



print("Total test accuracy: " + str(((found_accuracy + total_averaged_accuracy) * 100)/2) + "%")



#image = cv2.imread(PATH_TO_IMAGE)
#image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
#(boxes, scores, classes, num) = sess.run(
#    [detection_boxes, detection_scores, detection_classes, num_detections],
#    feed_dict={image_tensor: image_expanded})

#print(scores)

#myfile = open('elektro busilica aeg 2.xml', 'r')
#data=myfile.read()

#xml_object = objectify.fromstring(data)

#print(xml_object.object[0].name)
#print(len(xml_object.object))

#no_of_found = 0
#for i in range(int(num)):
#    if(scores[0][i] < 0.80):
#        break
#    no_of_found = no_of_found + 1
    
#partial_accuracy_correct_numbers = 1 - (abs(len(xml_object.object) - no_of_found) / len(xml_object.object))
#print (partial_accuracy_correct_numbers)



# All the results have been drawn on image. Now display the image.
#cv2.imshow('Object detector', image)

# Press any key to close the image
#cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()
