# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import csv
import os
#
def landmarks(image):

    image = image

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # load the input image, resize it, and convert it to grayscale

    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    landmarks=[]
    # loop over the face detections
    for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    for (x,y) in shape:
            landmarks.append((x,y))

    #print len(landmarks)
    print(1)
    return landmarks
    #print"Facial landmarks extraction of ",listofnames[j],"finished.."

# img = cv2.imread('C:/Users/wug/Desktop/project_gui_merge/real_time_img/0.jpg')
# cv2.imshow('aa',img)
# cv2.waitKey()
# print(len(landmarks(img)))
