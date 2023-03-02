import time
import win32gui
from ctypes import windll

import numpy
import pywintypes
from mss import mss

import functions
import image_transform

# Make program aware of DPI scaling
windll.user32.SetProcessDPIAware()

is_active_window = False


def grab_window():
    global is_active_window
    functions.controls_checker()
    while True:
        try:
            hwnd = win32gui.FindWindow("prism3d", None)
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        except pywintypes.error:
            print("ETS2 Window Not Found!")
            time.sleep(5)
        else:
            width = right - left
            height = bottom - top
            game_window = {'top': top, 'left': left, 'width': width, 'height': height}
            foreground_window_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            if foreground_window_name == "Euro Truck Simulator 2" or "American Truck Simulator":
                game_capture = numpy.array(mss().grab(game_window))
                is_active_window = True
                image_transform.image_manip(game_capture)
