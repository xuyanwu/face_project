# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import glob
import os
import cv2
import numpy as np
import math
import sys
#from avg import *
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
from loadOpenFace import face_feature,prepareOpenFace
import torch


face_cascade = cv2.CascadeClassifier('mallick_haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=480)

def real_time_recognition(frame,flag,model,face_features,face_names):


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    rects = detector(gray, 0)
    # count=0
    # Faces=-1
    # font = cv2.FONT_HERSHEY_SIMPLEX
    output = []
    output.append(np.ones((96,96,3),dtype=np.uint8))
    features = torch.zeros(1,128)
    if flag == 2 or flag == 3:
        for (i,rect) in enumerate(rects):
            if i ==0:
                output[0] = fa.align(frame, gray, rect)
                (x,y,w,h)= face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # text = "Hello Yvans!!"
                # cv2.putText(frame, text, (x, y-8), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                output.append(fa.align(frame, gray, rect))
                (x,y,w,h)= face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # text = "Hello Yvans!!"
                # cv2.putText(frame, text, (x, y-8), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if flag == 3:

            # print(output[0].shape)
            features = face_feature(model,output).data
            print(features.size())
            print(face_features.size())
            compare = torch.mm(features,torch.t(face_features))
            confidences,orders = torch.max(compare,1)
            confidences = confidences.cpu()
            orders = orders.cpu()
            # print(orders)
            for (i,rect) in enumerate(rects):
                (x,y,w,h)= face_utils.rect_to_bb(rect)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if confidences[i]<=0.5:
                    text = 'You are Mr.Nobody'
                else:
                    text = face_names[orders[i]]
                cv2.putText(frame, text, (x, y-8), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            features = features.cpu()


    return frame,features#,output

# model = prepareOpenFace().eval()
# frame = cv2.imread('aa.jpeg')
# flag = 3
# face_names,face_features = torch.load('../face_re_data/face_features.pt')
# face_features = face_features.cuda()
#
# sum,_,output = real_time_recognition(frame,flag,model,face_features,face_names)
#
# i = 0
# cv2.imshow('aa',sum)
# for img in output:
#     cv2.imshow(str(i),img)
#     i = i+1
# cv2.waitKey()






