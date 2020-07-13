import numpy as np
import cv2
from imutils import contours
from mss import mss
from PIL import Image
import time
from matplotlib import pyplot as plt

bounding_box = {'top': 80, 'left': 20, 'width': 580, 'height': 580}

sct = mss()


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


while True:

    # Getting the frame of capture
    sct_img = np.array(sct.grab(bounding_box))

    # Reducing the FPS
    #cv2.waitKey(20)

    # Converting the source image to Gray
    img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2LAB)


    ########### LIDANDO COM A FAIXA AMARELA AQUI ##########

    # Defining points to plot the transformation of the road
    pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

    # Points of the camera, on the road to Wrap
    # Bottom left | Bottom Right | Top Left | Top Right
    wrap_points = [(0, 470), (530, 420), (211, 273), (361, 273)]

    # Creating the perspective transformer, and using
    M = cv2.getPerspectiveTransform(np.float32(wrap_points), pts2)
    M_reverse = cv2.getPerspectiveTransform(pts2, np.float32(wrap_points))
    dst = cv2.warpPerspective(img, M, (400, 600))

    # Creating points on the source image, refering to the wrap points
    for points in wrap_points:
        cv2.circle(img, points, 3, (255, 0, 255), 3)

    # Applying Gausian filter to wrapped image and then canny algorithm
    # Gausianblur = cv2.GaussianBlur(dst, (3, 3), 1)
    # thresh, res = cv2.threshold(dst, 180, 300, cv2.THRESH_BINARY_INV)
    dst_yellow = cv2.split(dst)[2]
    res_yellow = dst_yellow
    #thresh, res_yellow = cv2.threshold(dst_yellow, 120, 250, cv2.THRESH_BINARY)
    #res_yellow = cv2.Canny(res_yellow, 50, 250)
    #res_yellow = cv2.GaussianBlur(res_yellow, (9, 9), 3)
    #res_yellow = res_yellow[:, :]

    dst_teste = cv2.warpPerspective(res_yellow, M_reverse, (400, 600))


    ####Teste#########
    dst_yellow[0:260,70:] = 0

    imgGray = dst_yellow

    #imgGray = cv2.warpPerspective(imgGray, M, (400, 600))
    #imgGray = cv2.addWeighted(imgGray, 1, imgGray, 0, -110)
    thresh, imgGray = cv2.threshold(imgGray, 120, 300, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3))
    imgBlur = cv2.GaussianBlur(imgGray, (21,21), 9)
    imgCanny = cv2.Canny(imgBlur, 50, 250)
    # imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, np.ones((10,10)))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=5)
    imgErode = cv2.erode(imgDial, kernel, iterations=5)

    imgColor = colorFilter(
        cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR))

    combinedImage = cv2.bitwise_or(imgColor, imgErode)

    imgErode2 = combinedImage


    ###########3

    res_yellow = cv2.bitwise_not(combinedImage)
    res_yellow = cv2.medianBlur(res_yellow, 9,9)



    ym_per_pix = 3 * 8 / 720  # meters per pixel in y dimension, 8 lines (5 spaces, 3 lines) at 10 ft each = 3m
    xm_per_pix = 3.7 / 550

    ### Settings
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 10

    # Take a histogram of the bottom half of the image
    histogram = np.sum(res_yellow[res_yellow.shape[0] // 2:, :], axis=0)

    # plt.figure()
    # plt.plot(histogram)

    # Create an output image to draw on and  visualize the result
    out_img2 = np.dstack((res_yellow, res_yellow, res_yellow)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] *0.70)
    leftx_base = np.argmax(histogram[:midpoint])
    # Set height of windows
    window_height = np.int(res_yellow.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = res_yellow.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = res_yellow.shape[0] - (window + 1) * window_height
        win_y_high = res_yellow.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img2, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)


        pts = np.array([[[win_xleft_low, win_y_low],
                         [win_xleft_high, win_y_high]]], dtype="float32")
        x, y = (cv2.perspectiveTransform(pts, M_reverse))[0][0]
        w, h = (cv2.perspectiveTransform(pts, M_reverse))[0][1]
        cv2.rectangle(sct_img, (x, y), (w, h), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))



    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]


    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    except:
        pass

    # Fit a second order polynomial to each


    #out_img2[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]





    ####################################################################3

    #img2 = cv2.cvtColor(sct_img, cv2.COLOR_BGR2HSV)
    #dst2 = cv2.warpPerspective(img2, M, (400, 600))

    imgGray = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.warpPerspective(imgGray, M, (400, 600))
    #imgGray = cv2.addWeighted(imgGray, 1, imgGray, 0, -110)
    thresh, imgGray = cv2.threshold(imgGray, 120, 250, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3))
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 250)
    # imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, np.ones((10,10)))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgErode = cv2.erode(imgDial,kernel,iterations=1)

    imgColor = colorFilter(cv2.warpPerspective(cv2.threshold(sct_img, 120, 300, cv2.THRESH_BINARY_INV)[1], M, (400, 600)))
    combinedImage = cv2.bitwise_or(imgColor, imgErode)

    imgErode = cv2.medianBlur(combinedImage, 21, 21)


    imgErode[0:240,0:290] = 0




    ### Settings
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 10

    # Take a histogram of the bottom half of the image
    histogram = np.sum(imgErode[imgErode.shape[0] // 2:, :], axis=0)

    # plt.figure()
    # plt.plot(histogram)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((imgErode,
                         imgErode,
                         imgErode)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    rightx_base = np.argmax(histogram[midpoint:])
    #
    # # Set height of windows
    window_height = np.int(imgErode.shape[0] / nwindows)
    # # Identify the x and y positions of all nonzero pixels in the image
    nonzero = imgErode.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # # Current positions to be updated for each window
    rightx_current = rightx_base
    # # Create empty lists to receive left and right lane pixel indices
    right_lane_inds = []
    #
    # # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = imgErode.shape[0] - (window + 1) * window_height
        win_y_high = imgErode.shape[0] - window * window_height
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image

        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(imgErode, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        pts = np.array([[[win_xright_low, win_y_low],
                         [win_xright_high, win_y_high]]], dtype="float32")
        x, y = (cv2.perspectiveTransform(pts, M_reverse))[0][0]
        w, h = (cv2.perspectiveTransform(pts, M_reverse))[0][1]
        cv2.rectangle(sct_img, (x, y), (w, h), (0, 255, 0), 2)


        # Identify the nonzero pixels in x and y within the window
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]


        # Append these indices to the lists
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_right_inds) >= minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            #print(rightx_current)


    # Concatenate the arrays of indices
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    right_fit =[0,0,0]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        pass

    # Fit a second order polynomial to each
    ym_per_pix = 3 * 8 / 720  # meters per pixel in y dimension, 8 lines (5 spaces, 3 lines) at 10 ft each = 3m
    xm_per_pix = 3.7 / 550
    right_fit_m = 0
    try:
        right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    except:
        pass

    tdfsdg = np.hstack([imgErode, cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)])
    tdfsdg2 = np.hstack([res_yellow, cv2.cvtColor(out_img2, cv2.COLOR_BGR2GRAY)])

    y_eval = ym_per_pix//1000
    curve_rad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    #print(curve_rad)




    # cv2.imshow('aa', np.hstack([out_img2, out_img]))

    cv2.imshow('aa',  sct_img)


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

