import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))
detector=HandDetector(maxHands=1)

offset=20
imgSize=300
counter=0

# save hand images
folder = "DATA/9"
os.makedirs(folder, exist_ok=True)

# save gesture videos
video_root = "VIDEO_DATA"
os.makedirs(video_root, exist_ok=True)

# word 
current_word = "Hello"
word_folder = os.path.join(video_root, current_word)
os.makedirs(word_folder, exist_ok=True)

recording = False
out = None
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = 20

video_writer = None  # Initialize video writer
recording_video = False  # Flag to check if we are recording video

while True:
    success, img=cap.read()
    if not success or img is None:
        print("Frame grab failed, skipping this loop...")
        continue   # go to next frame
    if recording and video_writer is not None:
        video_writer.write(img)

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

        else:
            # fix width of hand image
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))       
            imgResizeShape=imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :]=imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", img)

    # if recording, keep writing frames
    if recording and video_writer is not None:
        video_writer.write(img)

    key=cv2.waitKey(1)

    # save button (press "s" to record hand gesture)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{int(time.time())}.jpg', imgWhite)
        print(counter)
    
    # "v" starts/stops video recording
    elif key == ord("v"):
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_filename = f"{word_folder}/{current_word}_{int(time.time())}.avi"
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
            recording = True
            print(f" Recording started: {video_filename}")
        else:
            recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print(" Recording stopped and saved.")

    elif key == 27:   # press ESC to exit 
        # release if recording
        if video_writer is not None:
            video_writer.release()
        break


# cleanup
cap.release()
cv2.destroyAllWindows()
print("Program ended.")