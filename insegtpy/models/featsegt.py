#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:07:11 2021

@author: abda
"""

from insegtpy.models.kmdict import KMTree, DictionaryPropagator
from insegtpy.models.gaussfeat import GaussFeatureExtractor
import insegtpy.models.utils as utils
import insegtpy.models.segt as segt


class FeatSegt(segt.Segt):
    ''' 
    Segmentation using extracted features and KD trees.
    For now supports only Gausssian features, but can be made to use any kind
    of features.
    '''
    def __init__(self, image, 
                 branching_factor, number_layers, number_training_vectors, 
                 features_sigmas, propagation_size):
    # TODO, consider tree-branching, tree-layers, tree-training-size, propagation-lenght
        '''
        Initialization of GaussFeatSegt object.

        Parameters
        ----------
        image : numpy array
            2D image.
        branching_factor : integer
            Branching of kmtree.
        number_layers : integer
            Number of layeres in the kmtree.
        number_training_vectors : integer
            Number of features vectors extracted from image to train the 
            kmtree. If the number exceeds the total number of patches in the 
            image, then the training is done on all patches from the image. 
        features_sigmas : list of floats
             Standard deviation for extracted Gaussian features. An example 
             value is [1, 2, 4].
        features_normalize : boolean
            A flag inditacint that features are to be normalized according to 
            mean and standard deviation. Should normally be set to True.
        propagation_size : odd integer
            Side length of feature propagation step in pixels, for example 9.
            
        '''
        
        image = utils.normalize_to_float(image)
        
        self.feature_extractor = GaussFeatureExtractor(sigmas=features_sigmas) 
        features = self.feature_extractor(image, update_normalization=True)

        
        self.relator = KMTree(patch_size=1, branching_factor=branching_factor, 
                              number_layers=number_layers)
        self.relator.build(features, number_training_vectors)
        self.assignment = self.relator.search(features)
        self.propagator = DictionaryPropagator(self.relator.tree.shape[0], 
                                           patch_size=propagation_size)
        
    
    def process(self, labels, nr_classes=None):
        '''
        Update dictionary values and return probability image.
        
        Parameters
        ----------
        labels : 2D numpy array
            Label image of size (r,c) with values 0 (background) to l.

        Returns
        -------
        probabilities : 3D numpy array
            Probability image of size (l,r,c).

        '''

        labels_onehot = utils.labels_to_onehot(labels, nr_classes)
        self.propagator.improb_to_dictprob(self.assignment, labels_onehot)
        probabilities = self.propagator.dictprob_to_improb(self.assignment)
        return probabilities
    
  
    def segment_new(self, image):
        '''
        Compute the probability image for an new mage using the trained
        dictionary.

        Parameters
        ----------
        image : 2D numpy array
            Image of size (r,c) that should be processed. 

        Returns
        -------
        probabilities : numpy array
            Probability image of size (l,r,c), where l is the number of 
            segmentation labels.

        '''
        
        image = utils.normalize_to_float(image)
        features_this = self.feature_extractor(image, update_normalization=False)
        assignment_this = self.relator.search(features_this)
        probabilities = self.propagator.dictprob_to_improb(assignment_this)
        return utils.normalize_to_one(probabilities)
    
        

def gauss_features_segmentor(image,
              branching_factor = 5,  # dictionary
              number_layers = 5,  # dictionary
              number_training_vectors = 30000,  # dictionary
              features_sigma = [1,2,4],  # features
              propagation_size = 9, 
              scales = None,
              propagation_repetitions = None  # propagation
              ):
    ''' Convenience function to create segmentation model which uses Gauss 
    features and KM tree.
    
    '''
    
    if scales is None or scales==1 or scales==[1]:
        print('Bulding single-scale GaussFeatSegt model.')
        model = FeatSegt(image, branching_factor, 
                        number_layers, number_training_vectors,
                        features_sigma, propagation_size)
    else:
        print('Bulding multi-scale GaussFeatSegt model.')
        if type(scales) is not list: scales = [scales]
        
        model_init = lambda im: FeatSegt(im, branching_factor, 
                        number_layers, number_training_vectors,
                        features_sigma, propagation_size)
        model = segt.Multiscale(image, scales, model_init)
        
    if propagation_repetitions and propagation_repetitions>1:
        print('Adding propagation repetitions.')
        model = segt.Repeated(model, propagation_repetitions)
        
    return model

    
   


    


    