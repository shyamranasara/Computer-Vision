import cv2 as cv
import mediapipe as mp 
import time

mpHands = mp.solutions.hands
hands  = mpHands.Hands()
mpdrow = mp.solutions.drawing_utils

pTime = 0
cTime = 0


cap = cv.VideoCapture(0)

while True:
    success , img = cap.read()
    imageRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        # print(results)
        for handsLms in results.multi_hand_landmarks:
            # print(handsLms)
            for id , lm in enumerate(handsLms.landmark):
                # print(id , lm)
                h,w,c = img.shape
                cx , cy = int(lm.x *w), int(lm.y * h)
                print(id, cx , cy)
                if id == 4:
                    cv.circle(img, (cx, cy), 15,(255,0,255),cv.FILLED)
            
        mpdrow.draw_landmarks(img , handsLms , mpHands.HAND_CONNECTIONS)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img , str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,255),3)

        cv.imshow('Video', img)
        if cv.waitKey(20) & 0XFF == ord('q'):
            break
        

cap.release()
cv.destroyAllWindows()