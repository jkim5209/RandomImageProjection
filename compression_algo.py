#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:05:36 2020

@author: jaeykim
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import numpy as np
from scipy.stats import special_ortho_group

# img n dim column vector, basis k many n dim row orthogonal vectors (basis of subspace)
# Returns k many coeficients
def proj_img(img, basis):
    return np.matmul(img, basis)

# coef n dim row vector, basis k many n dim row orthogonal vectors (basis of subspace)
# Returns k many coeficients
def reconstruct(coef, basis):
    return np.matmul(basis, coef)




def main():
    # X = mpimg.imread("building.jpg")
    # rgb_weights = [0.2989, 0.5870, 0.1140]
    # gray_scale = np.dot(X[...,:3], rgb_weights)

    # plt.imsave("example.png", X)
    # plt.imsave("example_gray.png", gray_scale, cmap = cm.gray)
    
    #Y = special_ortho_group.rvs(dim= 256, size=1, random_state = 1)
    #print(Y)
    
    image_size = 10
    dim_proj = 7
    
    img = np.random.rand(image_size)
    img = img / np.linalg.norm(img, 2)
    
    Y = special_ortho_group.rvs(dim=image_size)
    
    basis = Y[:,range(dim_proj)]
    print("basis", basis)
    coef = proj_img(img, basis)
    
    img_reconstruct = reconstruct(coef, basis)
    print("Original image", img)
    print("Reconstructed image", img_reconstruct)
    print("MSE:", np.linalg.norm(img - img_reconstruct, 2) ** 2)
  
if __name__== "__main__":
  main()