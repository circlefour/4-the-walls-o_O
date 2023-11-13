import cv2 as cv
import numpy as np
import random

def get_start_end_dim(x0, x1):
    start = random.randint(x0, x1)
    end = random.randint(start, x1)
    dim = end-start
    return start, end, dim

def get_block_loc(x0, x1, dim):
    start = random.randint(x0, x1-dim)
    end = start+dim
    return start, end