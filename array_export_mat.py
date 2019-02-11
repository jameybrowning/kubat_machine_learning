# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:32:41 2018

@author: jb5182
"""
import numpy, scipy.io

arr = numpy.arange(9)
arr = arr.reshape((3, 3))  # 2d array of 3x3

scipy.io.savemat('C:/Users/jb5182/Google_Drive/Documents/ML/Codes/arrdata.mat', mdict={'arr': arr})
