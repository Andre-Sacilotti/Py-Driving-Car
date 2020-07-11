import numpy as np
import cv2
from mss import mss
from PIL import Image
import time

bounding_box = {'top': 80, 'left': 20, 'width': 580, 'height': 580}

sct = mss()
startTime = time.time()
while True:
    nowTime = time.time()
    sct_img = np.array(sct.grab(bounding_box))

    GRAY = cv2.cvtColor((sct_img), cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(GRAY, 120, 350)
    img = cv2.GaussianBlur(img,(3,3), 1)

    vertices= np.array([[0,600],[600,600],[500,200], [100,200]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), 255)
    img = cv2.bitwise_and(img, mask)

    lines = cv2.HoughLinesP(img, 1, np.pi/180, 180, 20, 15)
    try:
        for line in lines:

            d = (line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2
            if(d**(1/2) > 200):
                cv2.line(sct_img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255,255,0))
    except:
        pass


    cv2.imshow('screen2', (img))

    print(1/(nowTime-startTime))
    startTime = time.time()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

