import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model("Model/keras_model.h5")
classifier=Classifier("Model/keras_model.h5","Model/labels.txt")
labels=['A','B','C','D','E','F']

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

offset=20
imgSize=300
counter=0

# save hand images to this folder
folder = "DATA/F"

while True:
    success, img=cap.read()
    imgOutput = img.copy()
    hands, img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h= hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCrop = img[y-offset : y+h+offset , x-offset : x+w+offset]  
        imgCropShape=imgCrop.shape 
        #imgWhite[0:imgCropShape[0], 0:imgCropShape[1]]=imgCrop

        aspectRatio = h/w
        if aspectRatio>1:

            # fix height of hand image
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))       
            imgResizeShape=imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap]=imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)


        else:
            # fix width of hand image
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))       
            imgResizeShape=imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :]=imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

    # save button (press "s" to record hand gesture)
    