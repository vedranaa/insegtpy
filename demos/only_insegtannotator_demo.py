#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Aug 27 12:30:39 2021

An example on the use of InSegtAnnotator with the custom-made segmentation 
model. An image to be processed is an RGB image, but InSegtAnnotator needs to 
be given grayscale image for visualization. The segmentation model may still 
use RGB values of the image for processing.

@author: vand
'''


import numpy as np
import insegtpy
import skimage.data 

# image
image = skimage.data.astronaut()
image_gray = (255*skimage.color.rgb2gray(image)).astype(np.uint8)  


class Model():
    
    def process(self, labels):
        '''
        The simplest processing function for rgb images (computes a mean color for
            each label and assigns pixels to label with color closest to pixel color)
    
        Parameters:
            labels, 2D array with labels as uin8 (0 is background)
        Returns:
            segmentation, array of the same size and type as labels
        
        Author: vand@dtu.dk
    
        '''   
        N = labels.max()
        if not N:
            return labels
        # mean color for every label 1...N
        lable_colors = np.array([np.mean(image[labels==n], 0) 
                                      for n in range(1, N+1)])
        # pixel-to-color distances for all pixels and all labels
        layers = 1 if image.ndim==2 else image.shape[2]
        dist = ((image.reshape((-1 ,1, layers)) - 
                 lable_colors.reshape((1, N, layers)))**2).sum(axis=2)
        probs = np.exp(-dist/dist.max())
        probs /= probs.sum(axis=1, keepdims=True)
        
        return np.transpose(probs.reshape(labels.shape + (-1,)), (2,0,1))
        
 
insegtpy.insegt(image_gray, Model())

    
  