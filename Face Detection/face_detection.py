import cv2 as cv



def RescaleFrame(frame , scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width , height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)



img = cv.imread("mahakal_ujjain.jpg")
resized_img = RescaleFrame(img , 0.25)
cv.imshow("Shyam " , resized_img)


gray = cv.cvtColor(resized_img , cv.COLOR_BGR2GRAY)
cv.imshow("Gray Shyam", gray)

haar_coscade = cv.CascadeClassifier("haar_face.xml")
faces_rect = haar_coscade.detectMultiScale(gray , scaleFactor=1.6 , minNeighbors=3)
print("Number of Face Founded", len(faces_rect))


for (x,y,w,h) in faces_rect:
    cv.rectangle(resized_img ,(x,y),(x+w, y+h),(0,255,0),thickness=2)

cv.imshow("Detected Face ", resized_img)



cv.waitKey(0)