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


face_cascade = cv2.CascadeClassifier('mallick_haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=500)



def capture_face(frame):
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 600 pixels, and convert it to
    # grayscale
    frame = frame
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Calculate number of faces

    rects = detector(gray, 0)
    faceAligned = None
    face_exit = False
    for (i,rect) in enumerate(rects):
            face_exit = True
            faceAligned = fa.align(frame, gray, rect)

    return faceAligned, face_exit

