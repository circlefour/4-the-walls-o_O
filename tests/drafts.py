from glitchers import *
from img_processing import *
import cv2 as cv
import numpy as np
import time

def test_face_cage():
    m_face = np.zeros(shape=(235, 235, 3))
    face_cage(m_face)
    cv.waitKey()
    cv.destroyAllWindows()

# test_face_cage()

# import unittest

# class TestGlitchers(unittest.TestCase):
#     def test_face_cage(self):
#         m_face = np.zeros(shape=(135, 97, 3))
#         face_cage(m_face)
#         # and i don't know what to assert here

def scale_test_img(file_name='test_nature.png', scale_factor=5):
    img_test = cv.imread(file_name)
    return cv.resize(src=img_test, dsize=(img_test.shape[1]//scale_factor, img_test.shape[0]//scale_factor))

def night_vision():
    img_test = scale_test_img()
    hsv_img = cv.cvtColor(img_test, cv.COLOR_BGR2HSV)

    # thresholding
    lower = np.array([60, 35, 140])
    upper = np.array([180, 255, 255])
    mask = cv.inRange(hsv_img, lower, upper)

    masked_img = cv.bitwise_and(hsv_img, hsv_img, mask=mask)
    masked_or = cv.bitwise_or(img_test, hsv_img, mask=mask)


    cv.imshow('img', hsv_img)
    cv.imshow('masked and', masked_img)
    cv.imshow('back to rgb', cv.cvtColor(masked_img, cv.COLOR_HSV2BGR))
    cv.imshow('masked or', masked_or)
    cv.imshow('back to rgb2', cv.cvtColor(masked_or, cv.COLOR_HSV2BGR))
    cv.waitKey()
    cv.destroyAllWindows()

def or_with_flipped():
    img = scale_test_img()


def triangle_np(shape=(200, 150, 3)):
    img = np.zeros(shape)
    i = 0
    for h in range(0, shape[0]):
        for w in range(i, shape[1]):
            img[h, w] = [0,255,0]
        i+=1
    return img

def display(imgs):
    for img in imgs:
        cv.imshow(img['title'], img['image'])
    cv.waitKey()
    cv.destroyAllWindows()

def testing_axis_flips():
    img = triangle_np()
    imgs = []
    imgs.append({'image': img, 'title': 'original'})
    for i in range(1,4):
        title = 'img' + str(i-1)
        imgs.append({'image': np.flip(img, axis=i-1), 'title': title})
    imgs.append({'image': np.flip(img), 'title': 'img3'})
    display(imgs)

def bitwise_or_both_axes(img):
    # notted = cv.bitwise_not(img)
    # return cv.bitwise_or(img, notted)
    return cv.bitwise_or(img, np.flip(img))

# # night_vision()
# # testing_axis_flips()
# kewl = bitwise_or_both_axes(triangle_np())
# display([{'image': kewl, 'title': 'flip or'},
#         #  {'image': cool, 'title': 'cool'}
#          ])

# display([{'image': bitwise_or_both_axes(scale_test_img()), 'title': 'ayo'}])

# display([{'image': cv.bitwise_not(bitwise_or_both_axes(scale_test_img())), 'title': 'ayo'}])

# change format from CV_16U to CV_8U so i can use cvtColor
# scaled_image = cv.convertScaleAbs(triangle_np(), alpha=(255.0 / 65535.0))
# cv.imshow('green in hsv', cv.cvtColor(scaled_image, cv.COLOR_BGR2HSV))
# cv.waitKey()
# cv.destroyAllWindows()

def blue_green_bitwise(shape=(200, 200, 3)):
    blue = np.full(shape=shape, fill_value=[255, 0, 0], dtype='uint8')
    green = np.full(shape=shape, fill_value=[0, 255, 0], dtype='uint8')

    # AND
    bg = cv.bitwise_or(blue, green)
    bg2 = np.bitwise_or(blue, green)
    cv.imshow('blue AND green', bg)
    cv.imshow('NUMPY blue AND green', bg2)
    cv.waitKey()
    cv.destroyAllWindows()

def blue_green_addition(shape=(200, 200, 3)):
    blue = np.full(shape=shape, fill_value=[255, 0, 0], dtype='uint8')
    green = np.full(shape=shape, fill_value=[0, 255, 0], dtype='uint8')

    bg = blue + green
    cv.imshow('blue + green', bg)
    cv.waitKey()
    cv.destroyAllWindows()

def test_img_with_colour(shape=(200, 200, 3)):
    test_img = scale_test_img()
    test_img = np.resize(test_img, shape)

    blue = np.full(shape=shape, fill_value=[255, 0, 0], dtype='uint8')
    
    and_blue = cv.bitwise_or(blue, test_img)

    cv.imshow('original', test_img)
    cv.imshow('and', and_blue)
    cv.waitKey()
    cv.destroyAllWindows()

def test_crop_img_with_colour(shape=(200, 200, 3)):
    test_img = scale_test_img()
    test_img = test_img[0:shape[0], 0:shape[1]]

    blue = np.full(shape=shape, fill_value=[255, 0, 0], dtype='uint8')
    green = np.full(shape=shape, fill_value=[0, 255, 0], dtype='uint8')
    
    and_blue = cv.bitwise_and(blue, test_img)
    or_blue = cv.bitwise_or(blue, test_img)

    and_green = cv.bitwise_and(green, test_img)
    or_green = cv.bitwise_or(green, test_img)

    cv.imshow('original', test_img)
    # cv.imshow('and', and_blue)
    cv.imshow('and', and_green)
    cv.waitKey()
    cv.destroyAllWindows()

def bitwise():
    a = np.array([255, 100, 0])
    b = np.array([100, 25, 57])
    b = np.array([100, 25, 57])
    c = cv.bitwise_and(a, b)
    c = cv.bitwise_and(b, b)
    print(c)

def naive_night_vision():
    test_img = scale_test_img()
    green = np.full(shape=test_img.shape, fill_value=[0, 255, 0], dtype='uint8')
    and_green = cv.bitwise_and(green, test_img)
    cv.imshow('and', and_green)
    cv.waitKey()
    cv.destroyAllWindows()

def test_edge():
    img = cv.imread('one_face.jpg', cv.IMREAD_GRAYSCALE)
    img = edge_detect(img)
    cv.imshow('edges', img)
    cv.waitKey()
    cv.destroyAllWindows()


# blue_green_bitwise()
# blue_green_addition()
# test_img_with_colour()
# test_crop_img_with_colour()
# naive_night_vision()
test_edge()