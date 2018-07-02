import sys
import cv2
import os
import csv
import dlib
import imutils
import argparse
import numpy as np
from PyQt5.QtCore import *
from imutils.video import VideoStream
import math
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFontDialog, QStyleFactory, QAction, QMessageBox, QFileDialog)
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.uic import loadUi
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.uic import loadUi
from face_function.faceAverage import face_Average
from face_function.Real_Time import real_time_Average
from face_function.faceMorph import face_morphing
from face_function.landmarks_extraction1 import landmarks
from face_function.faceSwap import applyAffineTransform,warpTriangle,calculateDelaunayTriangles,rectContains
from face_function.capture_face import capture_face

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        # capture face image

        self.initial_database.clicked.connect(lambda:self.initial())

        self.start_cam_capture.clicked.connect(lambda:self.capture_face())
        self.start_capture.clicked.connect(lambda:self.capture_cam())
        self.save_capture.clicked.connect(lambda:self.capture_face_save())
        self.clear_capture.clicked.connect(lambda:self.capture_face_cancel())
        self.stop_cam_capture.clicked.connect(lambda:self.stop_capture_cam())
        #clear imgs
        self.clear_swap.clicked.connect(lambda:self.clear_swapping())
        self.clear_average.clicked.connect(lambda:self.clear_averaging())
        self.clear_morph.clicked.connect(lambda:self.clear_morphing())


        self.webcamStartBtn.clicked.connect(self.start_webcam)
        self.webcamStopBtn.clicked.connect(self.stop_webcam)
        self.eyesDetectBtn.setCheckable(True)
        self.eyesDetectBtn.toggled.connect(self.detect_webcam_eye)
        self.motionDetectBtn.setCheckable(True)
        self.motionDetectBtn.toggled.connect(self.detect_webcam_motion)
        self.motimgButton.clicked.connect(self.set_motion_image)
        self.faceAverageBtn.clicked.connect(self.set_face_average_image)
        self.realTimeFaceAverageBtn.clicked.connect(self.set_real_time_average_image)
        # Test for real time face average to start webcam
        #self.realTimeFaceAverageBtn.clicked.connect(self.start_webcam)
        self.faceSwapLoad1.clicked.connect(lambda: self.loadClicked(1))
        self.faceSwapLoad2.clicked.connect(lambda: self.loadClicked2(1))
        self.faceAverageLoad.clicked.connect(lambda: self.loadClicked2(31))
        self.stop_real_average_cam.clicked.connect(lambda: self.stop_real_averaging())

        self.faceSwapBtn.clicked.connect(self.face_swap_start)
        self.faceMorphLoad1.clicked.connect(lambda: self.loadClicked(41))
        self.faceMorphLoad2.clicked.connect(lambda: self.loadClicked2(42))
        self.faceMorphBtn.clicked.connect(self.perform_face_morph)

        self.image = None
        self.face_enabled = False
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.motion_enabled = False
        self.motionFrame = None
        self.face_average_enabled = False
        self.real_time_face_average_enabled = False
        self.path1 = None
        self.path2 = None
        self.key = 1
        self.average_img_num = 0
        self.morph_img1 = None
        self.morph_img2 = None
        self.swap_img1 = None
        self.swap_img2 = None
        self.capture_face_state = 0
        self.capture_face_num = 0
        # prepareOpenFace()


        quitAction = QAction('&Quit', self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.setStatusTip("Leave the app")
        quitAction.triggered.connect(self.close_application)

        helpAction = QAction("&Help", self)
        helpAction.setShortcut("Ctrl+H")
        helpAction.setStatusTip("Help")
        helpAction.triggered.connect(self.get_help)

        self.statusBar()

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(quitAction)

        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction(helpAction)

    #initial database
    def initial(self):
        path='./'
        for remove_file in os.listdir(path):
            print(1)
            if os.path.isfile(os.path.join(path,remove_file)) and '.jpg' in remove_file:
                # print(remove_file)
                os.remove(path + remove_file)
            QApplication.processEvents()


    #clear windows
    def clear_swapping(self):
        self.faceSwapLabel1.clear()
        self.faceSwapLabel2.clear()
        self.faceSwapOutputLabel.clear()

    def clear_averaging(self):
        self.faceAverageLbl1.clear()
        self.faceAverageLbl2.clear()

    def clear_morphing(self):
        self.faceMorphLabel1.clear()
        self.faceMorphLabel2.clear()
        self.faceMorphOutputLabel.clear()


    # capture imgs
    def capture_face_cancel(self):

        self.capture_face_state = 0

    def capture_face_save(self):

        self.capture_face_state = 1

    def capture_cam(self):

        self.capture_face_state = 2

    def stop_capture_cam(self):
        self.capture_face_state = 3

    def capture_face(self):
        vs = VideoStream(usePiCamera=-1 > 0).start()
        while True:
            face_exit = False

            frame = vs.read()
            self.displayImage2(frame,11)
            while (self.capture_face_state == 2 or self.capture_face_state == 1):
                face, face_exit = capture_face(frame)
                cv2.imwrite(str(self.capture_face_num)+'.jpg',face)
                # self.displayImage2(face,11)
                if self.capture_face_state == 1:
                    print(1)
                    self.capture_face_state = 0
                    if face_exit:
                        face_exit = False
                        cv2.imwrite(str(self.capture_face_num)+'.jpg',face)
                        face = None
                        self.capture_face_num = self.capture_face_num + 1
                        # QApplication.processEvents()

                QApplication.processEvents()
            if self.capture_face_state == 3:
                break
            QApplication.processEvents()
        vs.stop()
        self.capture_face_state = 0
        self.FaceCaptureLabel.clear()



    @pyqtSlot()
    # n here is to pass button information between face swap and face morphing
    def loadClicked(self, n):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open File' +
                                              "Image Files", '.',
                                              "(*.png *.jpg *.jpeg)")
        if fname:
            if n == 1:
                self.loadImage(fname, 1)
            elif n == 2:
                self.loadImage(fname, 3)
            elif n == 41:
                self.loadImage(fname, 41)
        else:
            print('Invalid file extension')


    # For face swap's second label
    def loadClicked2(self, n):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open File' +
                                              "Image Files", '.',
                                              "(*.png *.jpg *.jpeg)")
        if fname:
            if n == 1:
                self.loadImage(fname, 2)
            if n == 2:
                self.loadImage(fname, 4)
            elif n == 31:

                cv2.imwrite('./img/' + str(self.average_img_num) + '.jpg',cv2.imread(fname))
                self.average_img_num = self.average_img_num + 1
                self.loadImage(fname, 31)
            elif n == 42:
                self.loadImage(fname, 42)
        else:
            print('Invalid file extension')

    def loadImage(self, fname, my_path=0):
        if my_path == 1:
            self.path1 = fname
            self.image = cv2.imread(fname)
            self.swap_img1 = self.image
            self.displayImage(1)
        elif my_path == 2:
            self.path2 = fname
            self.image = cv2.imread(fname)
            self.swap_img2 = self.image
            self.displayImage(2)
        elif my_path == 3:
            self.path1 = fname
            self.image = cv2.imread(fname)
            self.displayImage(3)
        elif my_path == 4:
            self.path2 = fname
            self.image = cv2.imread(fname)
            self.displayImage(4)
        elif my_path == 31:
            self.path2 = fname
            self.image = cv2.imread(fname)
            self.displayImage2(self.image,31)
        elif my_path == 41:
            self.morph_img1 = cv2.imread(fname)
            self.displayImage2(self.morph_img1,41)
        elif my_path == 42:
            self.morph_img2 = cv2.imread(fname)
            self.displayImage2(self.morph_img2,42)
        else:
            print('FACE SWAP ERROR PATH')

    def displayImage(self, my_path):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:  # rows[0], cols[1], channels[2]
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0],
                     self.image.strides[0], qformat)
        # BGR ==> RGB
        img = img.rgbSwapped()
        if my_path == 1:
            self.faceSwapLabel1.setPixmap(QPixmap.fromImage(img))
            self.faceSwapLabel1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceSwapLabel1.setScaledContents(True)
        elif my_path == 2:
            self.faceSwapLabel2.setPixmap(QPixmap.fromImage(img))
            self.faceSwapLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceSwapLabel2.setScaledContents(True)
        elif my_path == 3:
            self.faceMorphLabel1.setPixmap(QPixmap.fromImage(img))
            self.faceMorphLabel1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceMorphLabel1.setScaledContents(True)
        elif my_path == 4:
            self.faceMorphLabel2.setPixmap(QPixmap.fromImage(img))
            self.faceMorphLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceMorphLabel2.setScaledContents(True)

    # Till here the functions are not in use ---------------------

    # Functions are in use from now on:
    def get_help(self):
        # Fill here later
        pass

    def detect_webcam_motion(self, status):
        if status:
            self.motion_enabled = True
            self.motionDetectBtn.setText('Detection Stop')
        else:
            self.motion_enabled = False
            self.motionDetectBtn.setText('Motion Detection')

    def set_motion_image(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.motionFrame = gray
        self.displayImage2(self.motionFrame, 2)

    # Face average
    def set_face_average_image(self):
        if self.average_img_num == 0:
            print('')
        else:
            self.average_img_num = 0
            average_img = self.start_face_average()
            self.displayImage2(average_img, 32)


    # Real time face average
    def keyPressEvent(self, event):
        # print("press：" + str(event.key()))
        if(event.key() == Qt.Key_Q):
            print('quit realtime_average')
            self.key=0

    def stop_real_averaging(self):
        self.key=0

    def set_real_time_average_image(self):
        self.key = 1
        vs = VideoStream(usePiCamera=-1 > 0).start()
        while True:

            frame = vs.read()
            frame , output = real_time_Average(frame)
            # cv2.imshow('output',output)
            # cv2.imshow("Frame", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #
    #         print(key)
    #
    # # if the `q` key was pressed, break from the loop
            if self.key == 0:

                break


            # cv2.imshow('aa',output)
            self.displayImage2(output, 32)
            self.displayImage2(frame, 31)
            QApplication.processEvents()
            # time.sleep(400)
        self.key=1
        vs.stop()
        time.sleep(4)
        path='./real_time_img/'
        for remove_file in os.listdir(path):
           if os.path.isfile(os.path.join(path,remove_file)) and '.jpg' in remove_file:
              # print(remove_file)
              os.remove(path + remove_file)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if self.face_enabled:
            detected_image = self.detect_eye(self.image)
            self.displayImage2(detected_image, 1)
        elif self.motion_enabled:
            detected_motion = self.detect_motion(self.image.copy())
            self.displayImage2(detected_motion, 1)
        else:
            self.displayImage2(self.image, 1)

    def detect_motion(self, input_img):
        self.text = 'No Motion'
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frameDiff = cv2.absdiff(self.motionFrame, gray)
        thresh = cv2.threshold(frameDiff, 40, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=5)

        im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        height, width, channels = input_img.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        for contour, hier in zip(cnts, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)

        if max_x - min_x > 80 and max_y - min_y > 80:
            cv2.rectangle(input_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            self.text = 'Motion Detected'

        cv2.putText(input_img, 'Motion Status: {}'.format(self.text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        return input_img

    def detect_eye(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))

        for x,y,w,h in faces:
            if self.faceCheckBox.isChecked():
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif self.eyesCheckBox.isChecked():
                ex, ey, ewidth, eheight = int(x + 0.125 * w), int(y + 0.25 * h), int(0.75 * w), int(
                    0.25 * h)

                cv2.rectangle(img, (ex, ey), (ex + ewidth, ey + eheight), (128, 255, 0), 2)

        return img

    def stop_webcam(self):
        self.timer.stop()
        self.capture.release()
        self.inputImgLabel.clear()
        self.videoFeedLabel.clear()

    def displayImage2(self, img, myWindow=1 ):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR ==> RGB
        outImage = outImage.rgbSwapped()

        if myWindow == 11:
            self.FaceCaptureLabel.setPixmap(QPixmap.fromImage(outImage))
            self.FaceCaptureLabel.setScaledContents(True)

        if myWindow == 1:
            self.videoFeedLabel.setPixmap(QPixmap.fromImage(outImage))
            self.videoFeedLabel.setScaledContents(True)
        if myWindow == 2:
            self.inputImgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.inputImgLabel.setScaledContents(True)
        # For face average
        if myWindow == 31:
            self.faceAverageLbl1.setPixmap(QPixmap.fromImage(outImage))
            #self.faceAverageLbl2.setScaledContents(True)   !Scaling reduces the quality!
        if myWindow == 32:
            self.faceAverageLbl2.setPixmap(QPixmap.fromImage(outImage))
        # For face swap
        if myWindow == 4:
            self.faceSwapOutputLabel.setPixmap(QPixmap.fromImage(outImage))
            self.faceSwapOutputLabel.setScaledContents(True)
        # For face morphing
        if myWindow == 41:
            self.faceMorphLabel1.setPixmap(QPixmap.fromImage(outImage))
            self.faceMorphLabel1.setScaledContents(True)
        if myWindow == 42:
            self.faceMorphLabel2.setPixmap(QPixmap.fromImage(outImage))
            self.faceMorphLabel2.setScaledContents(True)
        if myWindow == 5:
            self.faceMorphOutputLabel.setPixmap(QPixmap.fromImage(outImage))
            self.faceMorphOutputLabel.setScaledContents(True)

    def detect_webcam_eye(self, status):
        if status:
            self.eyesDetectBtn.setText('Stop Detection')
            self.face_enabled = True
        else:
            self.eyesDetectBtn.setText('Detect')
            self.face_enabled = False

    def close_application(self):
        choice = QMessageBox.question(self, 'Confirm Exit',
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Extracting Now")
            sys.exit()
        else:
            pass

    # Face Average functions --------------------
    def start_face_average(self):
        output = face_Average()

        return output
    # End of Face Average --------------

    # Real time face average fill it later

    # Face Swap----------------------


    def face_swap_start(self):



        img1 = self.swap_img1
        img2 = self.swap_img2
        # cv2.imshow('aa',img1)
        # cv2.imshow('bb',img2)
        cv2.waitKey()
        img1Warped = np.copy(img2)

        # Read array of corresponding points
        points1 = landmarks(img1)
        points2 = landmarks(img2)


        # Find convex hull
        hull1 = []
        hull2 = []

        hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

        for i in range(0, len(hullIndex)):
            hull1.append(points1[int(hullIndex[i])])
            hull2.append(points2[int(hullIndex[i])])

        # Find delaunay traingulation for convex hull points
        sizeImg2 = img2.shape
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = calculateDelaunayTriangles(rect, hull2)

        if len(dt) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            warpTriangle(img1, img1Warped, t1, t2)

        # Calculate Mask
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype)

        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        r = cv2.boundingRect(np.float32([hull2]))

        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

        self.displayImage2(output, 4)
    # End of face swap------------------

    # Face Morphing-----------------------

    # Warps and alpha blends triangular regions from img1 and img2 to morphing_img

    def face_morphing_start(self):
        # filename1 = os.path.basename(self.path1)
        # filename2 = os.path.basename(self.path2)

        img1 = self.morph_img1

        print(img1.shape)
        # cv2.imshow('aa',img1)
        img2 = self.morph_img2
        # cv2.imshow('bb',img2)
        # cv2.waitKey()
        landmarks1 = landmarks(img1)
        landmarks2 = landmarks(img2)
        morph = face_morphing(img1,landmarks1,img2,landmarks2)
        return morph

    def perform_face_morph(self):

        outputs = self.face_morphing_start()
        for output in outputs:
            self.displayImage2(output, 5)
            time.sleep(0.1)
            QApplication.processEvents()
