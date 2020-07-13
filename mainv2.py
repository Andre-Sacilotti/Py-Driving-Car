import math

import numpy as np
import cv2
from imutils import contours
from mss import mss
from PIL import Image
import time
from matplotlib import pyplot as plt

bounding_box = {'top': 80, 'left': 20, 'width': 580, 'height': 580}

# turn_points = [(405,379), (570,250),(575,0) , (320,260)]
# pts22 = np.float32([[0,0], [0, 600], [0,0], [600,600]])

wrap_points = [(142, 334), (414, 334), (208, 273), (354, 273)]
pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

img = None


sct = mss()
steps = 9
margem = 25
pixels = 50


#
# ColorFilter is a Function Based on Ross Kippenbrock
# That presented an seminar about lane finder in
# Pydata in Berlin
#
def colorFilter(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([18,94,140])
    upperYellow = np.array([48,255,255])
    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([255, 255, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    combinedImage = cv2.bitwise_or(maskedWhite,maskedYellow)
    return combinedImage




def left_lane_finder(img_left_lane):



    M = cv2.getPerspectiveTransform(np.float32(wrap_points), pts2)

    M_reverse = cv2.getPerspectiveTransform(pts2, np.float32(wrap_points))

    dst = cv2.warpPerspective(img_left_lane, M, (400, 600))

    dst= cv2.split(dst)[2]

    thresh, imgGray = cv2.threshold(dst, 120, 300, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3))
    imgBlur = cv2.GaussianBlur(imgGray, (21, 21), 9)
    imgCanny = cv2.Canny(imgBlur, 50, 250)
    # imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, np.ones((10,10)))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=5)
    imgErode = cv2.erode(imgDial, kernel, iterations=5)

    imgColor = colorFilter(
        cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR))

    combinedImage = cv2.bitwise_or(imgColor, imgErode)


    res_yellow = cv2.bitwise_not(combinedImage)
    dst = cv2.medianBlur(res_yellow, 9, 9)


    # histogram = np.sum(dst[dst.shape[0] // 2:, :], axis=0)
    #
    # out_img = np.dstack((dst, dst, dst)) * 255
    # midpoint = np.int(histogram.shape[0] / 2)
    # leftx_base = np.argmax(histogram[:midpoint])
    #
    # window_height = np.int(dst.shape[0] / steps)
    # nonzero = dst.nonzero()
    # nonzeroy = np.array(nonzero[0])
    # nonzerox = np.array(nonzero[1])
    # leftx_current = leftx_base
    # left_lane_inds = []
    #
    # for window in range(steps):
    #
    #     win_y_low = dst.shape[0] - (window + 1) * window_height
    #     win_y_high = dst.shape[0] - window * window_height
    #     win_xleft_low = leftx_current - margem
    #     win_xleft_high = leftx_current + margem
    #
    #     pts = np.array([[[win_xleft_low, win_y_low],
    #                      [win_xleft_high, win_y_high]]], dtype="float32")
    #     x, y = (cv2.perspectiveTransform(pts, M_reverse))[0][0]
    #     w, h = (cv2.perspectiveTransform(pts, M_reverse))[0][1]
    #     cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    #
    #     cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    #
    #     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
    #                 nonzerox < win_xleft_high)).nonzero()[0]
    #
    #     left_lane_inds.append(good_left_inds)
    #
    #     if len(good_left_inds) > pixels:
    #         leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    #
    # left_lane_inds = np.concatenate(left_lane_inds)
    #
    #
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    dst[:70,80:320] = 0
    dst = dst[:,0:np.int(dst.shape[1]*0.8)]

    dst2 = cv2.warpPerspective(img_left_lane, M, (400, 600))
    lines = cv2.HoughLinesP(dst, 1, np.pi/180, 25, 15)
    x1, x2, y1, y2 = (9999,9999,9999,9999)

    degreea = []
    linhasx = []
    linhasy = []

    try:
        for line in lines[:20]:
            vector = [line[0][2] - line[0][0], line[0][3] - line[0][1]]
            d = (vector[0]**2 + vector[1]**2)**(1/2)
            if d> 100:


                cv2.line(dst2, (line[0][0], line[0][1]),
                         (line[0][2], line[0][3]), (124,0,32), 2)

                pts = np.array([[[line[0][0], line[0][1]],
                                                      [line[0][2], line[0][3]]]], dtype="float32")
                x, y = (cv2.perspectiveTransform(pts, M_reverse))[0][0]
                w, h = (cv2.perspectiveTransform(pts, M_reverse))[0][1]

                if (y in linhasy) == False:
                    linhasx.append(x)
                    linhasy.append(y)

                cv2.line(img, (x, y), (w, h), (0, 255, 0), 2)



                vector2 = [600,0]
                sum = (vector[0]**2 + vector[1]**2)**(1/2) + (vector2[0]**2 + vector2[1]**2)**(1/2)
                degree = int(math.acos((vector[0] * vector2[0])/(sum)/180)*180/np.pi)



                degreea.append(degree)
    except:
        pass
    try:
        print("Left: " + str(np.polyfit((linhasx), (linhasy), 2)))
    except:
        pass

    # print("Left: " + str(np.mean(degreea)))

    return dst2


def right_lane_finder(img_right_lane):

    M = cv2.getPerspectiveTransform(np.float32(wrap_points), pts2)

    M_reverse = cv2.getPerspectiveTransform(pts2, np.float32(wrap_points))

    dst = cv2.warpPerspective(img_right_lane, M, (400, 600))

    thresh, imgGray = cv2.threshold(dst, 120, 250, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3))
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 250)
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)

    imgColor = colorFilter(
        cv2.warpPerspective(cv2.threshold(sct_img, 120, 300, cv2.THRESH_BINARY_INV)[1], M, (400, 600)))
    combinedImage = cv2.bitwise_or(imgColor, imgErode)

    imgErode = cv2.medianBlur(combinedImage, 21, 21)

    dst = imgErode

    # histogram = np.sum(img_right_lane[img_right_lane.shape[0] // 2:, :], axis=0)
    #
    # out_img = np.dstack((img_right_lane, img_right_lane, img_right_lane)) * 255
    #
    # midpoint = np.int(histogram.shape[0] / 2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #
    # window_height = np.int(img_right_lane.shape[0] / steps)
    #
    # nonzero = img.nonzero()
    # nonzeroy = np.array(nonzero[0])
    # nonzerox = np.array(nonzero[1])
    #
    # rightx_current = rightx_base
    #
    # right_lane_inds = []
    #
    #
    # for window in range(steps):
    #
    #     win_y_low = img_right_lane.shape[0] - (window + 1) * window_height
    #     win_y_high = img_right_lane.shape[0] - window * window_height
    #     win_xright_low = rightx_current - margem
    #     win_xright_high = rightx_current + margem
    #
    #
    #     pts = np.array([[[win_xright_low, win_y_low],
    #                      [win_xright_high, win_y_high]]], dtype="float32")
    #     x, y = (cv2.perspectiveTransform(pts, M_reverse))[0][0]
    #     w, h = (cv2.perspectiveTransform(pts, M_reverse))[0][1]
    #     cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    #
    #     cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    #
    #     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
    #                 nonzerox < win_xright_high)).nonzero()[0]
    #
    #     right_lane_inds.append(good_right_inds)
    #
    #
    #     if len(good_right_inds) > pixels:
    #         rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    #
    # # Concatenate the arrays of indices
    # right_lane_inds = np.concatenate(right_lane_inds)
    #
    # try:
    #     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # except:
    #     pass
    dst[:70, 80:320] = 0
    dst[:, :80] = 0

    dst2 = cv2.warpPerspective(img_right_lane, M, (400, 600))
    dst2[:, :80] = 0

    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 25, 15)


    degreea = []
    linhasx = []
    linhasy = []
    try:
        for line in lines[:20]:
            vector = [line[0][2] - line[0][0], line[0][3] - line[0][1]]
            d = (vector[0] ** 2 + vector[1] ** 2) ** (1 / 2)
            if d > 200:


                cv2.line(dst2, (line[0][0], line[0][1]),
                         (line[0][2], line[0][3]), (124, 0, 32), 2)

                pts = np.array([[[line[0][0], line[0][1]],
                                 [line[0][2], line[0][3]]]], dtype="float32")
                x, y = (cv2.perspectiveTransform(pts, M_reverse))[0][0]
                w, h = (cv2.perspectiveTransform(pts, M_reverse))[0][1]

                if (y in linhasy) == False:
                    linhasx.append(x)
                    linhasy.append(y)

                # cv2.line(img, (x, y), (w, h), (0, 255, 0), 2)
                cv2.circle(img, (x, y), 1, (255,0,155), 5)

                vector2 = [600, 0]
                sum = (vector[0] ** 2 + vector[1] ** 2) ** (1 / 2) + (vector2[0] ** 2 + vector2[1] ** 2) ** (1 / 2)
                degree = int(math.acos((vector[0] * vector2[0]) / (sum) / 180) * 180 / np.pi)

                degreea.append(degree)
    except:
        pass
    print("##############3")
    try:
        print("Right: " + str(np.polyfit((linhasx), (linhasy), 2)))
    except:
        pass





    # print("Right: " + str(np.mean(degreea)))


    return dst2


# def curve_vision_left(img_left_lane):
#
#     M = cv2.getPerspectiveTransform(np.float32(turn_points), pts22)
#
#     M_reverse = cv2.getPerspectiveTransform(pts22, np.float32(turn_points))
#
#     dst = cv2.warpPerspective(img_left_lane, M, (600, 600))
#
#     dst = cv2.split(dst)[2]
#
#     thresh, imgGray = cv2.threshold(dst, 130, 250, cv2.THRESH_BINARY_INV)
#
#     kernel = np.ones((3, 3))
#     imgBlur = cv2.GaussianBlur(imgGray, (21, 21), 9)
#     imgCanny = cv2.Canny(imgBlur, 50, 250)
#     imgDial = cv2.dilate(imgCanny, kernel, iterations=5)
#     imgErode = cv2.erode(imgDial, kernel, iterations=5)
#
#     imgColor = colorFilter(
#         cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR))
#
#     combinedImage = cv2.bitwise_or(imgColor, imgErode)
#
#
#     res_yellow = cv2.bitwise_not(combinedImage)
#     dst = cv2.medianBlur(res_yellow, 9, 9)
#
#     dst[140:,:] = 0
#
#     return dst

while True:

    sct_img = np.array(sct.grab(bounding_box))

    img = sct_img

    asfafs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_left_lane = cv2.cvtColor(sct_img, cv2.COLOR_BGR2LAB)
    img_right_lane = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)

    # for points in turn_points:
    #     cv2.circle(sct_img, points, 3, (255, 255, 0))

    for points in wrap_points:
        cv2.circle(img_left_lane, points, 3, (0, 255, 0))

    out_left_lane = left_lane_finder(img_left_lane)
    out_right_lane= right_lane_finder(img_right_lane)


    # out_left_curve = curve_vision_left(img_left_lane)

    cv2.imshow("aaaa", img)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
