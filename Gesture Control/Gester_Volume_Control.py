import cv2 as cv
import time
import numpy as np
import Hand_Tracking_Module as htm
import math


wCam , hCam = 640,500

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetetor(detectionCon=0.7)

while True:
    success , img = cap.read()

    img = detector.findHands(img=img)
    lmlist = detector.findPosition(img=img , draw=False)
    
    if len(lmlist)!=0:
        # print(lmlist[4], lmlist[8])
        x1, y1 = lmlist[4][1] , lmlist[4][2]
        x2 , y2 = lmlist[8][1] , lmlist[8][2]

        cx , cy = (x1+x2)//2 , (y1+y2)//2

        cv.circle(img,(x1 , y1), 
                  5,(255,0,255) , cv.FILLED)
        cv.circle(img , (x2 , y2), 5,(255,0,255),cv.FILLED)
        cv.line(img, (x1,y1),(x2 , y2),(255,0,255), 2)
        cv.circle(img ,(cx , cy ), 5, (255,0,255), cv.FILLED)

        length = math.hypot(x2 -x1 , y2 - y1)

        volBar = np.interp(length,[10,150],[400,150])
        volPar = np.interp(length , [10,150],[0,100])
        print(int(length))

        if length<50:
            cv.circle(img , (cx , cy), 5, (0,255,0), cv.FILLED)

        cv.rectangle(img , (50,150),(84,400),(0,255,0),1)
        cv.rectangle(img , (50, int(volBar)),(85,400),(0,255,0),cv.FILLED)

        cv.putText(img , f'{int(volPar)}%', (40,450),cv.FONT_HERSHEY_COMPLEX,0.5 , (0,250,0),2)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img , f'{int(fps)}',(40,50),cv.FONT_HERSHEY_COMPLEX ,0.5 ,(255,0,0),2)

        cv.imshow('Video', img)
        if cv.waitKey(20) & 0XFF == ord('q'):
            break


        