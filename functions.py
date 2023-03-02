import cv2
import numpy as np
import glob
import os
import ctypes
from pathlib import Path


def four_point_transform(capture):
    capture_width = capture.shape[1]
    capture_height = capture.shape[0]

    # roi values for 1300 x 1003 window
    # tl = (1059, 706)
    # tr = (1059+120, 706)
    # bl = (1059-20, 706+42)
    # br = (1059+140, 706+42)

    # roi scaled for current window size
    tl = (int(capture_width * (1059 / 1300)), int(capture_height * (706 / 1003)))
    tr = (int(capture_width * (1179 / 1300)), int(capture_height * (706 / 1003)))
    bl = (int(capture_width * (1039 / 1300)), int(capture_height * (44 / 59)))
    br = (int(capture_width * (1199 / 1300)), int(capture_height * (44 / 59)))

    srcpts = np.float32([[tl], [tr], [br], [bl]])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(srcpts, dst)
    warped = cv2.warpPerspective(capture, m, (max_width, max_height))
    # return the warped image
    return warped


def controls_checker():
    home = Path.home()

    # Define the path to the directory containing the profile
    directory = str(home) + r'\Documents\American Truck Simulator\profiles\*'

    # Use glob to get a list of all the folders in the directory
    folders = glob.glob(directory)

    # Use os.path.basename to get the name of each folder in the list
    folder_names = [os.path.basename(f) for f in folders]

    # Iterate over the list of folder names and print each one
    for name in folder_names:
        profile = directory.replace("*", "") + name
        controls_path = profile + r'\controls.sii'
        with open(controls_path) as f:
            file_contents = f.read()
        if 'di8' in file_contents or 'fusion' in file_contents:
            ctypes.windll.user32.MessageBoxW(0,
                                             "Warning: controls.sii using directinput, program may not work properly. "
                                             "If issues occur, change config_lines[0] to sys.keyboard, config_lines["
                                             "1] to sys.mouse",
                                             "ATSAutopilot Warning", 1)
