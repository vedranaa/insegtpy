#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 20:41:09 2021

@author: abda
"""

from insegtpy.models.kmdict import KMTree, DictionaryPropagator
from insegtpy.models.gaussfeat import get_gauss_feat_im
import insegtpy.models.utils as utils
import numpy as np
import time

# Building a multiscale dictionary from multiple images

# Compute Gaussian features
class GaussFeatureMultiIm:
    def __init__(self, sigmas = [1,2,4]):
        if type(sigmas) is not list: sigmas = [sigmas]
        self.sigmas = sigmas
        self.normalization_means = None
        self.normalization_stds = None

    def compute_features(self, images):
        '''
        Computes Gaussian features from a list of images and returns a list of 
        features

        Parameters
        ----------
        images : list of numpy arrays
            Input images.

        Returns
        -------
        feature_list : list of numpy arrays
            Output Gaussian features.

        '''
        feature_list = [] # Output list
        feat_sums = 0 # Statistics needed to compute normalization parameters
        feat_vars = 0
        feat_count = 0
        for i in range(len(images)):
            images[i] = utils.normalize_to_float(images[i])
            t = time.time()
            # Compute features.
            features = []
            for sigma in self.sigmas:
                features.append(get_gauss_feat_im(images[i], sigma))
            features = np.asarray(features).reshape((-1,)+images[i].shape)
            feature_list.append(features)
            if ( self.normalization_means is None ):
                n = np.prod(features.shape[1:])
                feat_sums += np.sum(features, axis=(1,2))
                feat_vars += (np.var(features, axis=(1,2))*n)
                feat_count += n
            # print(time.time()-t)
   
        if ( self.normalization_means is None):
            self.normalization_means = feat_sums/feat_count
            self.normalization_stds = (feat_vars/feat_count)**0.5
        
        # Normalize
        for i in range(len(feature_list)):
            feature_list[i] -= self.normalization_means.reshape((-1,1,1))
            feature_list[i] *= (1/self.normalization_stds).reshape((-1,1,1))
        return feature_list    

    def select_features(self, feature_list, n_feat_per_image = 10000):
        '''
        Select a random set of features from a list of feature images

        Parameters
        ----------
        feature_list : List of numpy arrays
            List of Gaussian features.
        n_feat_per_image : Integer, optional
            Number of features per image. If the image contains less features,
            the total number of features in the image will be used. The default 
            is 10000.

        Returns
        -------
        features_for_kmtree : Numpy array
            Array of features that can be used for KM-tree clustering.

        '''
        feature_dim = feature_list[0].shape[0]
        n_images = len(feature_list)
        features_for_kmtree = np.empty((feature_dim, n_images*n_feat_per_image, 1))
        
        t = 0
        for features in feature_list:
            n_feat_in_image = np.prod(features.shape[1:])
            n_feat = np.minimum(n_feat_in_image, n_feat_per_image)
            f = t # from index
            t += n_feat # to index
            random_idx = np.random.permutation(n_feat_in_image)[:n_feat]
            features_for_kmtree[:,f:t] = features.reshape((-1,n_feat_in_image))[:,random_idx].reshape((feature_dim,-1,1))
        return features_for_kmtree


# Class for computing multi-scale Gaussian features
class GaussFeatureScale:
    def __init__(self, scales = [1, 0.5, 0.25], sigmas = [1, 2, 4], n_feat_per_image = 10000):
        self.scales = scales
        self.sigmas = sigmas
        self.n_feat_per_image = n_feat_per_image
        self.gauss_feat_extractor = GaussFeatureMultiIm(sigmas)
    
    def compute_features(self, images):
        '''
        Computes features from a list of images

        Parameters
        ----------
        images : List of numpy arrays
            List of images.

        Returns
        -------
        features_list : List of list of numpy arrays
            Computes a list of Gaussian features at each scale that are put
            into a list.

        '''
        features_list = []
        for s in self.scales:
            image_list_scaled = [utils.imscale(im, scale=s) for im in images]
            feature_list_scaled = self.gauss_feat_extractor.compute_features(image_list_scaled)
            features_list.append(feature_list_scaled)
        return features_list

    def get_features_kmtree(self, features):
        '''
        Selects a list of features for building the KM-tree at each scale.

        Parameters
        ----------
        features : List of list of numpy arrays
            List of list of Gaussian features - one for each scale.

        Returns
        -------
        features_for_kmtree_list : List of numpy arrays
            List of Gaussian feautres for building the KM-tree.

        '''
        features_for_kmtree_list = []
        for feat in features:
            features_for_kmtree_list.append(
                self.gauss_feat_extractor.select_features(feat, self.n_feat_per_image))
        return features_for_kmtree_list
        
    def scale_labels(self, labels):
        '''
        Resizes label images based on scales.

        Parameters
        ----------
        labels : List of numpy arrays
            List of label images.

        Returns
        -------
        labels_list : List of list of numpy arrays
            Resize the list of label images that are put into a list - one for 
            each scale.

        '''
        labels_list = []
        for s in self.scales:
            labels_list.append([utils.imscale(l, scale = s, interpolation='nearest') for l in labels])
        return labels_list


# KM-tree at multiple scales
class KMTreeScale:
    def __init__(self, patch_size = 1, branching_factor = 5, number_layers = 5, 
                 normalization = False):
        self.patch_size = patch_size
        self.branching_factor = branching_factor
        self.number_layers = number_layers
        self.normalization = normalization
        self.km_trees = []
        
        
    def build_scale(self, features_for_kmtree):
        '''
        Computes multi-scale KM-tree.

        Parameters
        ----------
        features_for_kmtree : List of numpy arrays
            List of feature images.

        Returns
        -------
        None.

        '''
        for feat_km in features_for_kmtree:
            km_tree = KMTree(patch_size = self.patch_size, branching_factor = self.branching_factor, 
                              number_layers = self.number_layers, normalization = self.normalization)
            km_tree.build(feat_km, number_training_patches=feat_km.shape[1])
            self.km_trees.append(km_tree)
            
    def search_scale(self, features_list):
        '''
        Search multi-scale KM-tree.

        Parameters
        ----------
        features_list : List of list of numpy arrays.
            Feature images at multiple scales.

        Returns
        -------
        assignments_list : List of list of numpy arrays.
            Assignment images at multiple scales.

        '''
        assignments_list = []
        for feat, km_tree in zip(features_list, self.km_trees):
            assignments = [km_tree.search(f) for f in feat]
            assignments_list.append(assignments)
        return assignments_list


# Multi-scale computation of dictionary and image probabilities
class DictionaryPropagatorScale:
    def __init__(self, dictionary_sizes, patch_size = 15, nr_labels = None):
        self.patch_size = patch_size
        self.dictionary_sizes = dictionary_sizes
        self.propagators = [DictionaryPropagator(ds, patch_size) for ds in dictionary_sizes]
        self.nr_labels = nr_labels
    
    def improb_to_dictprob_labels(self, assignments_list, labels_list):
        '''
        Image probabilities ot dictionary probabilities. Sets the probabilities
        of the dictionaries.

        Parameters
        ----------
        assignments_list : List of list of numpy arrays
            Assignments at multiple scales.
        labels_list : List of list of numpy arrays
            Labels at multiple scales.

        Returns
        -------
        None.

        '''
        # for assignments, labels, i in zip(assignments_list, labels_list, range(len(self.propagators))):
        #     n_images = len(assignments)
        #     labels_onehot = utils.labels_to_onehot(labels[0])
        #     self.propagators[i].improb_to_dictprob(assignments[0], labels_onehot)
        #     probability_dictionary = self.propagators[i].probability_dictionary.copy()
        #     for label, assignment in zip(labels[1:], assignments[1:]):
        #         labels_onehot = utils.labels_to_onehot(label)
        #         self.propagators[i].improb_to_dictprob(assignment, labels_onehot)
        #         probability_dictionary += self.propagators[i].probability_dictionary
        #     self.propagators[i].probability_dictionary = probability_dictionary/float(n_images)

        for assignments, labels, propagator in zip(assignments_list, labels_list, self.propagators):
            n_images = len(assignments)
            labels_onehot = utils.labels_to_onehot(labels[0], nr_labels = self.nr_labels)
            propagator.improb_to_dictprob(assignments[0], labels_onehot)
            probability_dictionary = propagator.probability_dictionary.copy()
            for label, assignment in zip(labels[1:], assignments[1:]):
                labels_onehot = utils.labels_to_onehot(label, nr_labels = self.nr_labels)
                propagator.improb_to_dictprob(assignment, labels_onehot)
                probability_dictionary += propagator.probability_dictionary
            propagator.probability_dictionary = probability_dictionary/float(n_images)

    def improb_to_dictprob_scales(self, assignments, probabilities):
        '''
        Image probabilities to dictinoary probabilities for a single image. 
        Resets the dictionary probabilities.

        Parameters
        ----------
        assignments : List of numpy arrays
            Assignments for one image at multiple scales.
        probabilities : List of nupy arrays
            Probabilities of one image at multiple scales.

        Returns
        -------
        None.

        '''
        for assignment, probability, propagator in zip(assignments, probabilities, self.propagators):
            propagator.improb_to_dictprob(assignment, probability)
    
    def dictprob_to_improb_scales(self, assignments_list):
        '''
        Dictionary probabilities for multiple assignment images.

        Parameters
        ----------
        assignments_list : List of list of numpy arrays
            Assignments for multiple images at multiple scales.

        Returns
        -------
        probability_list : List of numpy arrays
            Probabilities for each image.

        '''
        probability_list = []
        for assignment in assignments_list[0]:
            probability_list.append(self.propagators[0].dictprob_to_improb(assignment))
        if ( len(assignments_list) > 1 ):
            for assignment_list, propagator in zip(assignments_list[1:], self.propagators[1:]):
                for assignment, i in zip(assignment_list, range(len(probability_list))):
                    probability_list[i] *= (utils.imscale(propagator.dictprob_to_improb(assignment), size = probability_list[i].shape[1:]))
        for i in range(len(probability_list)):
            # probability_list[i][probability_list[i] < 0] = 0
            # probability_list[i][probability_list[i] > 1] = 1
            p_sum = np.sum(probability_list[i], axis=0, keepdims = True)
            p_sum = p_sum + ( p_sum == 0 )
            probability_list[i] /= p_sum
        return probability_list
    

# Collection of functionality to build and optimize dictionaries at multiple scales
# using multiple images.
class GaussMultiImage:
    def __init__(self, scales = [1, 0.5, 0.25], 
                       sigmas = [1, 3, 8],
                       n_feat_per_image = 10000,
                       branching_factor = 5, 
                       number_layers = 5,
                       propagation_size = 9,
                       nr_labels = None
                       ):
        self.scales = scales
        self.sigmas = sigmas
        self.n_feat_per_image = n_feat_per_image
        self.branching_factor = branching_factor
        self.number_layers = number_layers
        self.propagation_size = propagation_size
        self.km_tree_scale = None
        self.assignments_list = None
        self.dict_prop_sc = None
        self.gfe_scale = None
        self.nr_labels = nr_labels
    
    def compute_dictionary(self, images, labels, return_probability = True):
        '''
        Extracts features, builds KM-trees, and computes probabilities from a
        set of multiple images and label images.

        Parameters
        ----------
        images : List of numpy arrays
            Images.
        labels : List of numpy arrays
            Labels.
        return_probability : Boolean, optional
            Determines if probabilities for images should be comoputed. 
            The default is True.

        Returns
        -------
        List of numpy arrays
            Probabilities of the training images. Will only return if the
            'return_probabilities' flag is set True.

        '''
        self.gfe_scale = GaussFeatureScale(scales = self.scales, sigmas = self.sigmas, 
                                      n_feat_per_image = self.n_feat_per_image)
        features_list = self.gfe_scale.compute_features(images)
        features_for_kmtree_list = self.gfe_scale.get_features_kmtree(features_list)
        labels_list = self.gfe_scale.scale_labels(labels)
    
        self.km_tree_scale = KMTreeScale(patch_size = 1, branching_factor = self.branching_factor, 
                                    number_layers = self.number_layers)
        self.km_tree_scale.build_scale(features_for_kmtree_list)
        self.assignments_list = self.km_tree_scale.search_scale(features_list)
        
        self.dict_prop_sc = DictionaryPropagatorScale(dictionary_sizes = 
                                                      [kmt.tree.shape[0] for kmt in self.km_tree_scale.km_trees], 
                                                      patch_size=self.propagation_size, nr_labels = self.nr_labels)
        self.dict_prop_sc.improb_to_dictprob_labels(self.assignments_list, labels_list)
        if return_probability:
            return self.dict_prop_sc.dictprob_to_improb_scales(self.assignments_list)
        
    def compute_probability(self, images):
        '''
        Computes the probabilities from a list of images.

        Parameters
        ----------
        images : List of numpy arrays
            Images.

        Returns
        -------
        List of numpy arrays
            Returns the comptued probabilities as a list.

        '''
        if ( self.dict_prop_sc is not None ):
            if type(images) is not list: images = [images]
            features = self.gfe_scale.compute_features(images)
            assignments = self.km_tree_scale.search_scale(features)
            return self.dict_prop_sc.dictprob_to_improb_scales(assignments)
    
    def segment_new(self, image):
        '''
        Segments a new image by computing the probability of a single image.

        Parameters
        ----------
        image : numpy arrays
            Image.

        Returns
        -------
        numpy arrays
            Returns the comptued probabilities as an image.

        '''
        return self.compute_probability(image)[0]
    
    def optimize(self, this_dict_propagator, assignments, residuals, alpha, beta):
        '''
        Optimizing the dictionary according to residuals.

        Parameters
        ----------
        this_dict_propagator : DictionaryPropagatorScale object 
            Object that allows for computing residual dictionary.
        assignments : List of numpy arrays
            Assignemnts for one image.
        residuals : List of numpy arrays
            Residual images.
        alpha : Float
            Determines how much the dictionary should be updated (learning rate).
        beta : Float
            Determines how much of a one-hot encoded image should be included
            in the probability image.

        Returns
        -------
        probability_new : numpy array
            Updated probability image.

        '''
        this_dict_propagator.improb_to_dictprob_scales(assignments, residuals) # computes the image probability
        # probability_new = np.ones(residuals[0].shape) # The output probability image
        for dict_prop, this_dict_prop, assignment in zip(self.dict_prop_sc.propagators, this_dict_propagator.propagators, assignments):            
            # Difference in probability dictionaries
            diff_prob_dict = dict_prop.probability_dictionary - this_dict_prop.probability_dictionary
            # Update the probability dictionary
            dict_prop.probability_dictionary = (dict_prop.probability_dictionary - alpha*diff_prob_dict)/(1 - alpha)
            # Normalize
            nelem_dict = dict_prop.probability_dictionary.shape[0]
            nelem_patch = self.propagation_size**2
            n_label = dict_prop.probability_dictionary.shape[1]//nelem_patch
            dict_sum = np.zeros((nelem_dict, nelem_patch))
            for i in range(n_label):
                dict_sum += dict_prop.probability_dictionary[:,i*nelem_patch:(i+1)*nelem_patch]
            dict_sum = np.tile(dict_sum, (1, n_label))
            dict_prop.probability_dictionary /= (dict_sum + 1e-1)
            dict_prop.probability_dictionary /= (1 - 1e-1)
        #     # Compute the probability at current scale
        #     probability = dict_prop.dictprob_to_improb(assignment)
        #     probability_max = getMaxP(probability)
        #     probability = beta*probability_max + (1-beta)*probability
        #     p_sum = np.sum(probability, axis=0, keepdims = True)
        #     p_sum = p_sum + ( p_sum == 0 )
        #     probability /= p_sum
        #     # Update new probability
        #     probability_new *= (utils.imscale(probability, size=residuals[0].shape[1:]) + 1e-10)
        # # Normalize probabilities
        # probability_new[probability_new < 0] = 0
        # probability_new[probability_new > 1] = 1
        # p_sum = np.sum(probability_new, axis=0, keepdims = True)
        # p_sum = p_sum + ( p_sum == 0 )
        # probability_new /= p_sum
        
        # return probability_new
    
    
    def optimize_constraint(self, this_dict_propagator, assignments, residuals, alpha, beta):
        '''
        Optimizing the dictionary according to residuals.

        Parameters
        ----------
        this_dict_propagator : DictionaryPropagatorScale object 
            Object that allows for computing residual dictionary.
        assignments : List of numpy arrays
            Assignemnts for one image.
        residuals : List of numpy arrays
            Residual images.
        alpha : Float
            Determines how much the dictionary should be updated (learning rate).
        beta : Float
            Determines how much of a one-hot encoded image should be included
            in the probability image.

        Returns
        -------
        probability_new : numpy array
            Updated probability image.

        '''
        this_dict_propagator.improb_to_dictprob_scales(assignments, residuals) # computes the image probability
        # probability_new = np.ones(residuals[0].shape) # The output probability image
        for dict_prop, this_dict_prop, assignment in zip(self.dict_prop_sc.propagators, this_dict_propagator.propagators, assignments):            
            # Difference in probability dictionaries
            diff_prob_dict = dict_prop.probability_dictionary - this_dict_prop.probability_dictionary
            # Update the probability dictionary
            dict_prop.probability_dictionary = (dict_prop.probability_dictionary - alpha*diff_prob_dict)/(1 - alpha)
            # Normalize
            nelem_dict = dict_prop.probability_dictionary.shape[0]
            nelem_patch = self.propagation_size**2
            n_label = dict_prop.probability_dictionary.shape[1]//nelem_patch
            dict_sum = np.zeros((nelem_dict, nelem_patch))
            for i in range(n_label):
                dict_sum += dict_prop.probability_dictionary[:,i*nelem_patch:(i+1)*nelem_patch]
            dict_sum = np.tile(dict_sum, (1, n_label))
            dict_prop.probability_dictionary /= (dict_sum + 1e-1)
            dict_prop.probability_dictionary /= (1 - 1e-1)
            dict_prop.probability_dictionary[dict_prop.probability_dictionary<0] = 0
            dict_prop.probability_dictionary[dict_prop.probability_dictionary>1] = 1
            dict_sum = np.zeros((nelem_dict, nelem_patch))
            for i in range(n_label):
                dict_sum += dict_prop.probability_dictionary[:,i*nelem_patch:(i+1)*nelem_patch]
            dict_sum = np.tile(dict_sum, (1, n_label))
            dict_prop.probability_dictionary /= (dict_sum + 1e-1)
            dict_prop.probability_dictionary /= (1 - 1e-1)
        
    
    def optimize_dictionaries(self, assignments_list, labels, probability_list = None, alpha = 0.01, 
                              beta = 0.5, n_iter = 1, verbose = True, noconstraint = True):
        '''
        Optimize dictionaries from assigments and labels

        Parameters
        ----------
        assignments_list : List of list of numpy arrays
            Assignments for multiple images at multiple scales.
        labels : List of numpy arrays
            List of label images.
        probability_list : List of numpy arrays, optional
            Probabilites for images. The default is None.
        alpha : Float, optional
            Determines how much the dictionary should be updated (learning rate). 
            The default is 0.01.
        beta : Float, optional
            Determines how much of a one-hot encoded image should be included
            in the probability image. The default is 0.5.
        n_iter : Integer, optional
            Number of iterations. The default is 1.
        verbose : Boolean, optional
            If learnign accuracy should be printed. The default is True.

        Returns
        -------
        probability_list : List of numpy arrays
            Probabilities for training images.

        '''
        # if probability_list is None: # if probability_list is not given, then it will be computed
        #     probability_list = self.dict_prop_sc.dictprob_to_improb_scales(assignments_list)
        
        this_dict_propagator = DictionaryPropagatorScale(dictionary_sizes = 
                                                      [kmt.tree.shape[0] for kmt in self.km_tree_scale.km_trees], 
                                                      patch_size=self.propagation_size, nr_labels = self.nr_labels)            
        
        for k in range(n_iter): # iterate
            perf = 0
            ct = 0
            idx = np.random.permutation(len(assignments_list[0]))
            for i in idx:
                assignments = []
                for j in range(len(assignments_list)):
                    assignments.append(assignments_list[j][i])
                probability = self.dict_prop_sc.dictprob_to_improb_scales([assignments])[0]
                # combine with max probability
                probability_max = getMaxP(probability)
                probability = beta*probability_max + (1-beta)*probability
                p_sum = np.sum(probability, axis=0, keepdims = True)
                p_sum = p_sum + ( p_sum == 0 )
                probability /= p_sum
                residual = utils.labels_to_onehot(labels[i], nr_labels = self.nr_labels) - probability
                # residual = utils.labels_to_onehot(labels[i]) - probability_list[i]
                
                residuals = []
                for s in self.scales:
                    residuals.append(utils.imscale(residual, scale = s))
                if noconstraint:
                    self.optimize(this_dict_propagator, assignments, residuals, alpha = alpha, beta = beta)
                else:
                    self.optimize_constraint(this_dict_propagator, assignments, residuals, alpha = alpha, beta = beta)
                # probability_list[i] = self.optimize(this_dict_propagator, assignments, residuals, alpha = alpha, beta = beta)
                if verbose:
                    probability = self.dict_prop_sc.dictprob_to_improb_scales([assignments])
                    labels_predict = utils.segment_probabilities(probability[0])
                    perf += np.sum((labels_predict - labels[i])==0)/(np.prod(labels[i].shape))
                    ct += 1
                    if ( ct % 1 == 0 ):
                        print(f'Iteration {k+1} of {n_iter} sub {ct} of {len(assignments_list[0])} accuracy {perf/ct:0.5f}')
            print(f'Accuracy {perf/ct:0.5f}')
        # return probability_list


def getMaxP(P):
    '''
    Computes a row-by-col-by-l image where the layer with the maximum value is
    set to 1.

    Parameters
    ----------
    P : numpy array
        Probability image.

    Returns
    -------
    Pmax : numpy array
        Image of maximum probability.

    '''
    Lmax = np.argmax(P, axis = 2)
    Pmax = P*0
    for i in range(0,P.shape[2]):
        Pmax[Lmax==i,i] = 1
    return Pmax
