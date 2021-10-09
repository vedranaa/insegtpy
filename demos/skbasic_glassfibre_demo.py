#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo showing how insegt may be used with basic insegt model for interactive
segmentation. This model relies on scikit-learn python package, so it may be
used if cpp code is misbihaving. As examples we use CT image of glass fibres.

Created on Fri Oct 16 00:08:59 2020
@author: vand
"""

import insegtpy
import insegtpy.models
import skimage.io


image = skimage.io.imread('../data/glass.png') 

# single-scale model
model = insegtpy.models.sk_basic(image, 
                              patch_size=9, 
                              nr_training_patches=10000, 
                              nr_clusters=100,
                              propagation_repetitions=2)
insegtpy.insegt(image, model)

