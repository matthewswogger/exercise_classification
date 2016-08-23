import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture('video/2.mp4')
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    edges = cv2.Canny(img,100,100)

    r = 400.0 / edges.shape[1]
    dim = (400, int(edges.shape[0] * r))
    resized = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('edges', resized)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
