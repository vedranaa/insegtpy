#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 08:27:35 2021

@author: abda, vand
"""

import insegtpy
import insegtpy.models
import PIL
import numpy as np
import matplotlib.pyplot as plt

image = np.array(PIL.Image.open('../data/NT2_0001.png'))
model = insegtpy.models.gauss_features_segmentor(image, 
                                   branching_factor = 12, 
                                   number_layers = 4,
                                   number_training_vectors = 100000,
                                   features_sigma = [1,2,4,8],
                                   propagation_size = 15, 
                                   scales=[1, 0.7, 0.5])

#Now you can choose ONE of the three ways of using the model: A, B, or C.

#%% A: Interactive use
ex = insegtpy.insegt(image, model)
      #  ... interaction
seg = insegtpy.utils.segment_probabilities(ex.probabilities) 

#%% B: Interactive starting with pre-prepared labeling
labels = np.array(PIL.Image.open('../data/NT2_0001_labels.png'))
ex = insegtpy.insegt(image, model, labels=labels)
      #  ... interaction
seg = insegtpy.utils.segment_probabilities(ex.probabilities) 


#%% C: Non-interactive use
labels = np.array(PIL.Image.open('../data/NT2_0001_labels.png'))
seg = insegtpy.utils.segment_probabilities(model.process(labels))


#%% Testing on another image

image_new = np.array(PIL.Image.open('../data/NT2_0512.png'))
prob_new = model.segment_new(image_new)
seg_new = insegtpy.utils.segment_probabilities(prob_new)
fig, ax = plt.subplots(2, 2, sharex = True, sharey = True )
ax[0][0].imshow(image)
ax[1][0].imshow(seg)
ax[1][0].set_title('Trained')
ax[0][1].imshow(image_new)
ax[1][1].imshow(seg_new)
ax[1][1].set_title('New')

















