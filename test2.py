import cv2
import cvzone
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import math
cap = cv2.VideoCapture(0)
classifier=Classifier("keras_model.h5","labels.txt")
detector2=FaceMeshDetector(maxFaces=1)
idList=[22,23,24,26,110,157,158,159,160,161,130,243]
ratioList=[]
counter=0
count=0
imgSize=300
offset=20
#hand detector
detector=HandDetector(maxHands=1,detectionCon=0.8)

labels=["A","B","C","D","E","F"]
ans2=""
ans=""
ans3=""
while True :
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success , img =cap.read()
    hands, img = detector.findHands(img)
    img, faces = detector2.findFaceMesh(img,draw=False)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop =img[y-offset:y+h+offset,x-offset :x+w+offset]

        imgCropShape=imgCrop.shape
        imgWhite[0:imgCropShape[0] , 0:imgCropShape[1]] = imgCrop
        aspectRatio=h/w
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize=cv2.resize(imgCrop , (wCal , imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize
            prediction,index=classifier.getPrediction(imgWhite)
            ans+=labels[index]



        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            ans += labels[index]

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
        if faces:
            face = faces[0]
            for id in idList:
                cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)
            leftup = face[159]
            leftdown = face[23]
            leftleft = face[130]
            leftright = face[243]
            length, _ = detector.findDistance(leftup, leftdown)
            length2, _ = detector.findDistance(leftleft, leftright)
            cv2.line(img, leftup, leftdown, (0, 200, 0), 3)
            cv2.line(img, leftleft, leftright, (0, 200, 0), 3)
            l = (length / length2) * 100
            ratioList.append(l)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)
            if ratioAvg < 35 and count == 0:
                counter +=1
                ans2 = ans[len(ans) - 1]
                ans3 += ans2
                print(ans3)
                ans2 = ""
                count = 1
            if count != 0:
                count += 1
                if count > 10:
                    count = 0

            cvzone.putTextRect(img, f'blink count:{counter}', (50, 100))

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("p"):
        ans3+=" "
    if key == ord("r"):
        print(ans3.split())
        text_to_speech = pyttsx3.init()
        text_to_speech.say(ans3.split())
        text_to_speech.runAndWait()

