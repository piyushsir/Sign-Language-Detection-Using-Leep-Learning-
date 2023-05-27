import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
#width,height=1280,720
#cap.set(3,width)
#cap.set(4,height)

imgSize=300
offset=20
folder = "data/A"
counter=0
#hand detector
detector=HandDetector(maxHands=1,detectionCon=0.8)
while True :
    success , img =cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop =img[y-offset:y+h+offset,x-offset :x+w+offset]

        imgCropShape=imgCrop.shape
        imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop
        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[ hGap:hCal + hGap,:] = imgResize




        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if key == ord("s"):
        counter+=1;
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)


