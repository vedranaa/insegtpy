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

image = skimage.io.imread('../data/NT2_0512.png') 



model = insegtpy.models.gauss_features_segmentor(image, 
                                   branching_factor = 25, 
                                   number_layers = 3,
                                   features_sigma = [1,2,4,8], 
                                   number_training_patches = 40000,
                                   propagation_patch_size = 9, 
                                   propagation_repetitions=1,
                                   scales=[1])

ex = insegtpy.insegt(image, model)



#%% Testing on another image
im_test = skimage.io.imread('../data/NT2_0512.png')
prob_test = model.new_image_to_prob(im_test)

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True )
ax[0][0].imshow(image)
ax[1][0].imshow(ex.probabilities[1])
ax[1][0].set_title('Train')
ax[0][1].imshow(im_test)
ax[1][1].imshow(prob_test[1])
ax[1][1].set_title('Test')

















