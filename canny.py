import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    edges = cv2.Canny(img,100,100)
    cv2.imshow('edges', edges)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()