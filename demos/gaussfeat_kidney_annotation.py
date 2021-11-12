#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:43:18 2021

@author: abda, vand
"""

import insegtpy
import insegtpy.models
import skimage.io
import matplotlib.pyplot as plt
import PIL
import numpy as np

#%% Load image from tiff stack
image_train = (np.minimum(skimage.io.imread('/Users/abda/Documents/Center/QIM/data/Rat_kidney/'
            'Rat_37_zoom/RAT 37 - 20-05-21 - RAT 38 - 20-05-21_RAT 38-40kV-5microA_recon.tiff')[:,400,:]/255,255)).astype(np.uint8) # change data type from uint16 to uint8

plt.figure()
plt.imshow(image_train, cmap='gray')

#%% Build the model

model = insegtpy.models.gauss_features_segmentor(image_train, 
                                   branching_factor = 25, 
                                   number_layers = 3,
                                   number_training_vectors = 200000,
                                   features_sigma = [1,2,4,8],
                                   propagation_size = 9, 
                                   scales=[1, 0.5, 0.25])

#%% Train on already annotated image

labels = np.array(PIL.Image.open('/Users/abda/Documents/Center/QIM/data/Rat_kidney/Rat_37_zoom/training/annotations_index.png'))//30
plt.figure()
plt.imshow(labels)

model.process(labels)


#%% Compute probabilities for the training image and the test image and display

prob_image_train = model.segment_new(image_train)
seg_image_train = insegtpy.utils.segment_probabilities(prob_image_train)

image_test = (np.minimum(skimage.io.imread('/Users/abda/Documents/Center/QIM/data/Rat_kidney/'
            'Rat_37_zoom/RAT 37 - 20-05-21 - RAT 38 - 20-05-21_RAT 38-40kV-5microA_recon.tiff')[:,660,:]/255,255)).astype(np.uint8) # change data type from uint16 to uint8



prob_image_test = model.segment_new(image_test)
seg_image_test = insegtpy.utils.segment_probabilities(prob_image_test)

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True )
ax[0][0].imshow(image_train)
ax[1][0].imshow(seg_image_train)
ax[1][0].set_title('Train')
ax[0][1].imshow(image_test)
ax[1][1].imshow(seg_image_test)
ax[1][1].set_title('Test')


#%% Interactive training

# Data can be saved by pressing 's'. The annotator will save data to the 
# current working directory. Instead the annotations can also be saved by 
# running the script below.

model.propagation_repetitions = 2

ex = insegtpy.insegt(image_train, model) # Opens a window for annotation
ex.saveAddress = '/Users/abda/Documents/Center/QIM/data/Rat_kidney/Rat_20/Zoom/training/'
#%% Save the data from the interactive segmentation

labels = ex.rgbaToLabels(ex.pixmapToArray(ex.annotationPix))

plt.figure()
plt.imshow(labels)


#%%
ex = insegtpy.insegt(image_train, model, labels) # Opens a window for annotation





























