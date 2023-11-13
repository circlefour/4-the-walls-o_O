import cv2 as cv
import numpy as np
import random

def edge_detect(img):
    edges = cv.Canny(img, 100, 200)
    return edges

# channels
def one_to_three(img):
    img2 = np.zeros((img.shape[0], img.shape[1], 3))
    img2[:,:,0] = img
    img2[:,:,1] = img
    img2[:,:,2] = img
    return img2