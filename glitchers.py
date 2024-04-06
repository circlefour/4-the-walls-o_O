import cv2 as cv
import numpy as np
import random
from glitcher_helpers import *
from scipy.ndimage.filters import gaussian_filter

# FIXME: default numpy sort function goes too hard : make custom one
def pxsort_frame(frame, axis=0):
    return np.sort(frame, axis=axis)

def on_start():
    print("press esc to end this madness")
    # FIXME: user input to change effect you want

# FIXME: have number of iterations be dependent on face size within frame : or some percentage of screen size : far away faces look blurry anyway
def block_glitch(frame, iterations=40, trippy=False):
    h, w, c = frame.shape
    if not trippy:
        glitched_face = frame.copy()
    else:
        glitched_face = frame
    for i in range(iterations):
        ystart, yend, ydim = get_start_end_dim(0, h)
        xstart, xend, xdim = get_start_end_dim(0, w)
        new_ystart, new_yend = get_block_loc(0, h, ydim)
        new_xstart, new_xend = get_block_loc(0, w, xdim)

        try:
            glitched_face[ystart:ystart+ydim, xstart:xstart+xdim] = frame[new_ystart:new_ystart+ydim, new_xstart:new_xstart+xdim]
            # super ultra glitchy
            # glitched_face[ystart:ystart+ydim, xstart:xstart+xdim] = glitched_face[new_ystart:new_ystart+ydim, new_xstart:new_xstart+xdim]
        except:
            continue
    return glitched_face

def pixelation(frame, px=10):
    face_roi = frame
    roi_height, roi_width, _ = face_roi.shape
   
    w = roi_width//px
    h = roi_height//px

    if w == 0 or h == 0:
        px = 1
        w = roi_width
        h = roi_height
            
    # FIXME: do this without loops : with some numpy functions prolly
    for i in range(px):
        for j in range(px):
            block = face_roi[i*h : i*h + h, j*w : j*w + w]
            if np.any(np.isnan(block)) or np.any(np.isinf(block)):
                continue
            face_roi[i*h : i*h + h, j*w : j*w + w] = np.mean(block, axis=(0,1))
    return frame

def gaussian_blur(frame):
    return gaussian_filter(frame, sigma=7)

# martine ali baguette bag inspired
# also like spy movies where you need to manuever around lazers in a hallway so you can get the diamond or recover some clue that leads you to the evil scientists lair so you can stop their plan and also rescue your friend
# but also this would look cooler around the entire thing, and not just on someones face
# it's stupid having default values because this will look very different at different distances
def face_cage(face, wlines=10, hlines=10):
    # if I don't copy it, lines start appearing on the actual thing aka messes up the array for other functions that use just_face
    # face = face.copy()
    h, w, channels = face.shape
    hdist = h // hlines
    wdist = w // wlines
    hcurr = 0
    wcurr = 0
    while (hcurr < h):
        # BGR colour scheme
        cv.line(face, (0, hcurr), (w, hcurr), (0,255,0), thickness=1)
        hcurr += hdist
    while (wcurr < w):
        cv.line(face, (wcurr, 0), (wcurr, h), (0,255,0), thickness=1)
        wcurr += wdist
    return face

# experimentation
def linear_combination_of_frames(frame):
    something = cv.bitwise_not(cv.bitwise_or(frame, np.flip(frame, axis=2)))
    something_else = cv.bitwise_or(frame, np.flip(frame, axis=0))
    return something * 2 + something_else

def rand_linear_combination_of_frames(frame, min=0, max=10):
    factor = random.randint(min,max)
    something = cv.bitwise_not(cv.bitwise_or(frame, np.flip(frame, axis=2)))
    something_else = cv.bitwise_or(frame, np.flip(frame, axis=0))
    return something * factor + something_else

def rand_linear_frames(frame, factor=1):
    something = cv.bitwise_not(cv.bitwise_or(frame, np.flip(frame, axis=2)))
    something_else = cv.bitwise_or(frame, np.flip(frame, axis=0))
    return something * factor + something_else

def first(frame):
    something = cv.bitwise_not(cv.bitwise_or(frame, np.flip(frame, axis=2)))
    something_else = cv.bitwise_or(frame, np.flip(frame, axis=0))
    # return something + something_else
    return something * something_else

def glitched_grey(frame):
    frame = frame + np.flip(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame
