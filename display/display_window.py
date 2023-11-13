"""
the display window
"""

import cv2 as cv

class DisplayWindow:
    def __init__(self, config, window_name):
        self.config = config
        self.window_name = window_name
        self.window_created = False

    def create_window(self):
        cv.namedWindow(self.window_name)
        cv.resizeWindow(self.window_name, self.config.width, self.config.height)
        self.window_created = True

    def display_frame(self, frame):
        if not self.window_created:
            return
        try:
            cv.imshow(self.window_name, frame)
        except cv.error as e:
            print(e)

    def __del__(self):
        cv.destroyWindow(self.window_name)