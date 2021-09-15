#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:07:11 2021

@author: abda
"""

import numpy as np
from insegtpy.models.kmdict import KMTree, DictionaryPropagator
import insegtpy.models.utils as utils
from insegtpy.models.gaussfeat import GaussFeatureExtractor
import skimage.transform


class OnescaleGaussFeatSegt:
    # Class for single-scale dictionary-based segmentation using Gaussian 
    # features and KD trees.
    def __init__(self, image, 
                 branching_factor, number_layers, number_training_vectors, 
                 features_sigmas, features_normalize, 
                 propagation_size, propagation_repetitions):
    # TODO, consider tree-branching, tree-layers, tree-training-size, propabation-lenght
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
        propagation_repetitions : integer
            Number of feature propagation steps. Normally 1 or 2.
            
        '''
        
        image = utils.normalize_to_float(image)
        
        self.feature_extractor = GaussFeatureExtractor(sigmas=features_sigmas) 
        features = self.feature_extractor(image, 
                                    update_normalization=features_normalize)

        
        self.relator = KMTree(patch_size=1, branching_factor=branching_factor, 
                              number_layers=number_layers)
        self.relator.build(features, number_training_vectors)
        self.assignment = self.relator.search(features)
        self.propagator = DictionaryPropagator(self.relator.tree.shape[0], 
                                           patch_size=propagation_size)
        self.propagation_repetitions = propagation_repetitions
        
    
    def single_update(self, labels_onehot):
        '''
        Single update stap taking (onehot) labels, updating values of the 
        dictionary and returning image probabilities.
        
        Parameters
        ----------
        labels_onehot : 3D numpy array
            Onehot label image of size (l,r,c) where l is the number of labels.
            Unlabeled pixels have zeros in all labels.

        Returns
        -------
        probabilities : 3D numpy array
            Probability image of the same size as labels_onehot.

        '''

        self.propagator.improb_to_dictprob(self.assignment, labels_onehot)
        probabilities = self.propagator.dictprob_to_improb(self.assignment)
        return probabilities
    
    def process(self, labels):
        '''
        Processing used by insegt annotator when working in the interactive 
        mode. Calls single_udpate a number of times.

        Parameters
        ----------
        labels : 2D numpy array
            Label image of size (r,c) with values 0 (background) to l.

        Returns
        -------
        probabilities : 3D numpy array
            Probability image of size (l,r,c).
            
        '''
                
        labels_onehot = utils.labels_to_onehot(labels)
        probabilities = self.single_update(labels_onehot)
               
        # If nr_repetitions>1, repeat
        for k in range(self.propagation_repetitions - 1):
            labels = utils.segment_probabilities(probabilities)
            labels_onehot = utils.labels_to_onehot(labels)
            probabilities = self.single_update(labels_onehot)  
        
        return utils.normalize_probabilities(probabilities)
        

    def new_image_to_prob(self, image):
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
        return utils.normalize_probabilities(probabilities)
    
    
class GaussFeatSegt:
    # Class for multi-scale dictionary-based segmentation using Gaussian 
    # features and KD trees. Uses single-scale class.
    
    # TODO, use another package (PIL or cv2) for scaling
    def __init__(self, image, scales,
                 branching_factor, number_layers, number_training_vectors, 
                 features_sigmas, features_normalize, 
                 propagation_size, propagation_repetitions):
        '''
        Initialization of GaussFeatSegt object.

        Parameters
        ----------
        image : numpy array
            2D image.
        scales : list of floats
           Downscaling factors for multi-scale feature extraction.
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
        propagation_repetitions : integer
            Number of feature propagation steps. Normally 1 or 2.
            
        '''
        
        image = utils.normalize_to_float(image)
        self.scales = scales
        self.propagation_repetitions = propagation_repetitions
        self.segt_list = [OnescaleGaussFeatSegt(skimage.transform.rescale(
                                        image, scale, preserve_range=True),      
                                        branching_factor, number_layers, 
                                        number_training_vectors, 
                                        features_sigmas, features_normalize,
                                        propagation_size, 1)
                          for scale in self.scales]
        self.aditive_parameter = 1e-10
       
            
    def single_update(self, labels_onehot):
        '''
        Single update stap taking (onehot) labels, updating values of the 
        dictionary and returning image probabilities.
        
        Parameters
        ----------
        labels_onehot : 3D numpy array
            Onehot label image of size (l,r,c) where l is the number of labels.
            Unlabeled pixels have zeros in all labels.

        Returns
        -------
        probabilities : 3D numpy array
            Probability image of the same size as labels_onehot.

        '''
        
        probabilities = np.ones(labels_onehot.shape)
        if probabilities.shape[0]>0:  # rescale breaks when empty image
            for scale, segt in zip(self.scales, self.segt_list):
                
                labels_onehot_sc = skimage.transform.rescale(labels_onehot.transpose((1,2,0)),
                            scale, order=0, multichannel=True, anti_aliasing=False,
                            preserve_range = True).transpose((2,0,1))
                
                probabilities_sc = segt.single_update(labels_onehot_sc) 
                probabilities *= (skimage.transform.resize(probabilities_sc, 
                            labels_onehot.shape, preserve_range=True) + 
                                  self.aditive_parameter)
        return utils.normalize_probabilities(probabilities)

    def process(self, labels):
        '''
        Processing used by insegt annotator when working in the interactive 
        mode. Calls single_udpate a number of times.

        Parameters
        ----------
        labels : 2D numpy array
            Label image of size (r,c) with values 0 (background) to l.

        Returns
        -------
        probabilities : 3D numpy array
            Probability image of size (l,r,c).
            
        '''
        # TODO This is a copy of process from GaussFeatSegt. Can we simplify code? 

        labels_onehot = utils.labels_to_onehot(labels)
        probabilities = self.single_update(labels_onehot)
               
        # If nr_repetitions>1, repeat
        for k in range(self.propagation_repetitions - 1):
            labels = utils.segment_probabilities(probabilities)
            labels_onehot = utils.labels_to_onehot(labels)
            probabilities = self.single_update(labels_onehot)  
        
        return utils.normalize_probabilities(probabilities)

    
    def new_image_to_prob(self, image):
        '''
        Compute the probability image for an new mage using the trained
        multi-scale dictionary.

        Parameters
        ----------
        image : 2D numpy array
            Image of size (r,c) that should be processed. Note that the image 
            will be resized for feature extraction, so make sure that the data 
            type and range of this image is the same as the training image(s).

        Returns
        -------
        probabilities : numpy array
            Probability image of size (l,r,c), where l is the number of 
            segmentation labels.

        '''
        image = utils.normalize_to_float(image)
        probabilities = np.array([[[1.0]]])  # needs to be 3D
        
        for scale, segt in zip(self.scales, self.segt_list):
            image_sc = skimage.transform.rescale(image, scale, preserve_range = True)
            probabilities_sc = segt.new_image_to_prob(image_sc)
            probabilities = probabilities * (skimage.transform.resize(probabilities_sc, 
                              (probabilities_sc.shape[0],) + image.shape, 
                              preserve_range=True) + self.aditive_parameter)           
        return utils.normalize_probabilities(probabilities)
    
    

def gauss_features_segmentor(image,
              branching_factor = 5,  # dictionary
              number_layers = 5,  # dictionary
              number_training_vectors = 30000,  # dictionary
              features_sigma = [1,2,4],  # features
              propagation_size = 9, 
              propagation_repetitions = 1,  # propagation
              scales = None
              ):
    ''' Convenience function for create segmentation model which uses
    (multiscale) Gauss features and KM tree.
    
    TODO parameters and return
    
    '''
    
    if scales is None:
        model = OnescaleGaussFeatSegt(image, 
                        branching_factor, number_layers, number_training_vectors,
                        features_sigma, True,  
                        propagation_size, propagation_repetitions)
    else:
        if type(scales) is not list: scales = [scales]
        model = GaussFeatSegt(image, scales,
                        branching_factor, number_layers, number_training_vectors,
                        features_sigma, True,  
                        propagation_size, propagation_repetitions)
    return model


    


    