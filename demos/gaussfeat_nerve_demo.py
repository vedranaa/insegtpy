#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 08:27:35 2021

@author: abda, vand
"""

import insegtpy
import insegtpy.models
import skimage.io
import matplotlib.pyplot as plt

image0 = skimage.io.imread('../data/NT2_0001.png') 


model = insegtpy.models.gauss_features_segmentor(image0, 
                                   branching_factor = 25, 
                                   number_layers = 3,
                                   number_training_vectors = 40000,
                                   features_sigma = [1,2,4,8],
                                   propagation_size = 9, 
                                   scales=[1, 0.5, 0.25])

ex = insegtpy.insegt(image0, model)


#%% Testing on another image

prob_image0 = ex.probabilities
seg_image0 = insegtpy.utils.segment_probabilities(prob_image0)


image1 = skimage.io.imread('../data/NT2_0512.png')
prob_image1 = model.new_image_to_prob(image1)
seg_image1 = insegtpy.utils.segment_probabilities(prob_image1)

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True )
ax[0][0].imshow(image0)
ax[1][0].imshow(seg_image0)
ax[1][0].set_title('Train')
ax[0][1].imshow(image1)
ax[1][1].imshow(seg_image1)
ax[1][1].set_title('Test')

















