import cv2 as cv
import mediapipe as mp
import time

class handDetetor():
    def __init__(self, node=False , maxHands =2 , detectionCon = 0.5 , trackCon = 0.5):
        self.node = node
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDrow = mp.solutions.drawing_utils


    def findHands(self , img , draw = True):
        imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDrow.draw_landmarks(img , handLms, self.mpHands.HAND_CONNECTIONS)

        return img
        

    def findPosition(self , img , handsNo=0 , draw = True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHands = self.results.multi_hand_landmarks[handsNo]

            for id, lm in enumerate(myHands.landmark):
                h , w, c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img , (cx, cy), 15 , (255,0,255),cv.FILLED)
        return lmList
    
def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetetor()
    while True:
        sucess , img = cap.read()
        img = detector.findHands(img )
        lmList = detector.findPosition(img,draw=False)

        if len(lmList)!= 0:
            print(lmList[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
            
        cv.putText(img , str(int(fps)), (10,70),cv.FONT_HERSHEY_COMPLEX, 1, (255,0,255),3)

        cv.imshow('Image', img)
        cv.waitKey(1)


if __name__ == '__main__':
    main()