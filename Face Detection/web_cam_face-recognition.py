import cv2 as cv
import numpy as np

video_capture =cv.VideoCapture(0)

while True:
    isTrue , frame = video_capture.read()
    # cv.imshow("Video " ,frame)

    rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    gray = cv.cvtColor(rgb_frame , cv.COLOR_BGR2GRAY)
    # cv.imshow("Gray Shyam", gray)

    haar_coscade = cv.CascadeClassifier("haar_face.xml")
    faces_rect = haar_coscade.detectMultiScale(gray , scaleFactor=1.8, minNeighbors=3)
    print("Number of Face Founded", len(faces_rect))


    for (x,y,w,h) in faces_rect:
        cv.rectangle(rgb_frame ,(x,y),(x+w, y+h),(0,255,0), thickness=2)

    cv.imshow("OUTPUT" , rgb_frame)

    if cv.waitKey(200) & 0xFF==ord("d"):
       break

video_capture.release()
cv.destroyAllWindows()