import random
import time

import cv2

from functions import *
from Line import Line
import pyautogui
import pydirectinput
import grabwindow

vehicle_offsets_window = []


def image_manip(capture):
    # ROI after perspective transform
    warped = four_point_transform(capture)

    # increase window size, helps with visualization and when I was messing around with small ROIs
    # this technically could introduce aliasing, which is bad
    # TODO: Only scale the output images, probably no need to scale capture (depending on ROI?)
    scale_percent = 200  # percent of original size
    width = int(warped.shape[1] * scale_percent / 100)
    height = int(warped.shape[0] * scale_percent / 100)
    dim = (width, height)
    scaled_warped = cv2.resize(warped, dim, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(scaled_warped, cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(hsv, 11)

    # mask for red path/road to follow
    path_lower = np.array([0, 143, 152])
    path_upper = np.array([10, 255, 255])
    path_mask = cv2.inRange(blur, path_lower, path_upper)
    # closes any small gaps in mask
    merge = cv2.morphologyEx(path_mask, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))

    # this fills in the arrows that obscure the red path
    # TODO: with findContours() we may not need to worry about filling in red path
    arrow_lower = np.array([32, 83, 74])
    arrow_upper = np.array([84, 255, 255])
    arrow_mask = cv2.inRange(blur, arrow_lower, arrow_upper)

    # this fills in the arrows, making the path (in theory) uniformly filled
    combined_mask = merge + arrow_mask

    # closes any small gaps
    merge2 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))

    # TODO: technically canny on binary image is wrong, findCountours() might be better
    # canny edge detection, not ideal, can have problems if path is obscured
    canny = cv2.Canny(merge2, 254, 255)

    detect_lines(canny, scaled_warped, capture)


def detect_lines(canny, orig_warped, capture):
    window_size = 5
    left_line = Line(n=window_size)
    right_line = Line(n=window_size)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(canny[canny.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = int(canny.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = canny.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # the width of the windows +/- margin
    margin = 20

    # minimum number of pixels found to recenter window
    minpix = 1

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = canny.shape[0] - (window + 1) * window_height
        win_y_high = canny.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # win_xright_low = rightx_current - margin - 10
        # win_xright_high = rightx_current + margin - 10
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(orig_warped, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 1)
        cv2.rectangle(orig_warped, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 50, 0), 1)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean pos
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # DEBUG: enough points found?
    # min_inds = 10
    # if leftx.size < min_inds or rightx.size < min_inds:
    #    print("not enough points found ", "leftx ", str(leftx.size), "rightx ", str(rightx.size))

    if not len(leftx) is 0:
        left_fit = np.polyfit(lefty, leftx, 2)

    if not len(rightx) is 0:
        right_fit = np.polyfit(righty, rightx, 2)

    if not len(leftx) is 0:
        left_fit = left_line.add_fit(left_fit)

    if not len(rightx) is 0:
        right_fit = right_line.add_fit(right_fit)

    if (len(leftx) is not 0) and (len(rightx) is not 0):
        # Generate x and y values for plotting
        ploty = np.linspace(0, orig_warped.shape[0] - 1, orig_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw the lines on
        color_warp = np.zeros((canny.shape[0], canny.shape[1], 3), dtype='uint8')

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))

        bottom_y = color_warp.shape[0] - 1
        # Calculate left and right line positions at the bottom of the image
        bottom_x_left = left_fit[0] * (bottom_y ** 2) + left_fit[1] * bottom_y + left_fit[2]
        bottom_x_right = right_fit[0] * (bottom_y ** 2) + right_fit[1] * bottom_y + right_fit[2]

        offset = orig_warped.shape[1] / 2 - (bottom_x_left + bottom_x_right) / 2
        lane_width = (bottom_x_right - bottom_x_left) / 2
        vehicle_offset = round((offset * 100 / lane_width), 2)

        # moving average of offset
        # TODO: a better way to ignore anomalous values
        moving_average_size = 10
        if len(vehicle_offsets_window) < moving_average_size:
            vehicle_offsets_window.append(vehicle_offset)
        else:
            del vehicle_offsets_window[0]

        # Calculate the average of current window
        offset_average = sum(vehicle_offsets_window) / len(vehicle_offsets_window)

        cv2.putText(color_warp, str(offset_average), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(color_warp, str(vehicle_offset), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Create an image to draw on and an image to show the selection window
        out_img = (np.dstack((canny, canny, canny)) * 255).astype(np.uint8)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        show_windows(color_warp, result, orig_warped, canny)
        move(offset_average, capture)


def show_windows(color_warp, result, orig_warped, canny):
    cv2.imshow('window', orig_warped)
    cv2.imshow('window2', canny)
    cv2.imshow('window3', color_warp)
    cv2.imshow('window4', result)

    cv2.moveWindow("window", 20, 220)
    cv2.moveWindow("window2", 20, 420)
    cv2.moveWindow("window3", 20, 620)
    cv2.moveWindow("window4", 20, 820)
    cv2.waitKey(1)


def move(offset, capture):
    dist_from_center = offset
    width = capture.shape[1]
    height = capture.shape[0]

    # TODO: Active Window Detection not working
    # if grabwindow.is_active_window:
    #     if dist_from_center > 10:
    #         # turn left
    #         # TODO: SCALE BASED ON DEVIATION
    #         pydirectinput.keyDown('a')
    #         time.sleep(0.001)
    #         pydirectinput.keyUp('a')
    #     if dist_from_center < -10:
    #         # turn right
    #         # TODO: SCALE BASED ON DEVIATION
    #         pydirectinput.keyDown('d')
    #         time.sleep(0.001)
    #         pydirectinput.keyUp('d')
    #     if -5 < dist_from_center < 5:
    #         pydirectinput.keyUp('a')
    #         pydirectinput.keyUp('d')
