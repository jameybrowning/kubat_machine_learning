# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:32:41 2018

@author: jb5182
"""


import scipy.io
import numpy as np
# Load characters 1-9 dataset
from sklearn import datasets
digits = datasets.load_digits()
image_arr = digits.images
target = digits.target


scipy.io.savemat('C:/Users/jb5182/Google_Drive/Documents/ML/Codes/image_arr.mat', mdict={'image_arr': image_arr})
scipy.io.savemat('C:/Users/jb5182/Google_Drive/Documents/ML/Codes/target.mat', mdict={'target': target})
