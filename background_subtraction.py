import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video/2.mp4')
# fgbg = cv2.BackgroundSubtractorMOG()


while(1):
    ret, frame = cap.read()
    # fgmask = fgbg.apply(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    r = 400.0 / thresh.shape[1]
    dim = (400, int(thresh.shape[0] * r))
    resized = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('frame',resized)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
