import numpy as np
import cv2



s = cv2.VideoCapture(0)
b = 64.  # brightness
c = 0.  # contrast
while True:
    ret, img1 = s.read()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    img = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    cv2.imshow("img", img1)
    img2 = cv2.addWeighted(img1, 1. + c / 127., img, 0, b - c)
    cv2.imshow("img1", img2)
    cv2.waitKey(10)