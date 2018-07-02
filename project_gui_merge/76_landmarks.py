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
path='./morphing_img/'
listofnames = []
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and '.jpg' and '.txt' not in i:
        listofnames.append(i)
print ("There's ",len(listofnames),"pictures in the Gallery")
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# load the input image, resize it, and convert it to grayscale

for j in range(len(listofnames)):
    
    image = cv2.imread(str(path+listofnames[j]))
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w,h=np.shape(gray)

     
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
            #landmark.append([x,y])
    print (len(landmarks))
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);
    lands=np.vstack([landmarks,boundaryPts])
    print (len(lands))
    path1='./morphing_img/'
    resultFile = open(os.path.join(path1+listofnames[j]+'.txt'),'w',newline='')
    wr = csv.writer(resultFile)
    for line in landmarks:
        print(line)
        wr.writerow(line)
    resultFile.close()
    #print"Facial landmarks extraction of ",listofnames[j],"finished.."
