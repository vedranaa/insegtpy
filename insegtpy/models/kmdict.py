#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:53:41 2021

@author: abda
"""
import ctypes
from glob import glob
import numpy.ctypeslib as ctl
import numpy as np
import os
from os.path import join
import platform

# Load library
if platform.system() == 'Windows':
    _libfile_ext = '.pyd'
else:
    _libfile_ext = '.so'
libfile = glob(join(os.path.dirname(__file__),  'km_dict.*' + _libfile_ext))
if len(libfile) == 0:
    raise FileNotFoundError(f'Could not find library file')
elif len(libfile) > 1:
    raise FileNotFoundError(f'Found more than one library file')
libfile = libfile[0]
lib = ctypes.cdll.LoadLibrary(libfile)

class KMTree:
    # Class for creating the relations in the image, i.e. building the km-tree
    # and searching the km-tree.
    def __init__(self, patch_size = 15, branching_factor = 5, number_layers = 5,
                 normalization = False):
        '''
        Initialize the Relate object.

        Parameters
        ----------
        patch_size : integer, optional
            Side length of the image patch. Should be an odd number.
            The default is 15.
        branching_factor : integer, optional
            Branching of kmtree.
        number_layers : integer, optional
            Number of layeres in the kmtree. The default is 5.
        normalization : Boolean, optional
            Normalize patches to unit length. The default is False.

        Returns
        -------
        None.

        '''
        self.patch_size = patch_size
        self.branching_factor = branching_factor
        self.number_layers = number_layers
        self.normalization = normalization
        self.tree = None


    def build(self, image, number_training_patches = 30000):
        '''
        Builds a k-means search tree from image patches. Image patches of size
        patch_size are extacted, and the tree is build by hierarchical k-means where
        k is the branching_factor. To reduce processing time, the tree is build from
        number_training_patches. If this exceeds the the total number of patches in
        the image, then the trainin is done on all possible patches from the image.
        The resulting kmeans tree is a 2D array.

        Parameters
        ----------
        image : numpy array
            2D (rows, cols), 3D (channels, rows, cols) or
            4D (n_im, channels, rows, cols) image
        number_training_patches : integer
            Number of patches used for training the kmtree. If the number exceeds
            the total number of patches in the image, then the trainin is done on
            all possible patches from the image.

        Returns
        -------
        None

        '''
        # Check patch size input
        if ( self.patch_size < 0 or self.patch_size%2 != 1 ):
            print('Patch size must be positive and odd!')
            return -1

        # python function for building km_tree
        py_km_tree_multi = lib.build_km_tree_multi
        # say which inputs the function expects
        py_km_tree_multi.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_bool,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_km_tree_multi.restype = None

        # Check image dimensions
        if image.ndim==2:
            rows, cols = image.shape[0:2]
            channels = 1
            n_im = 1
        elif image.ndim==3:
            channels, rows, cols = image.shape
            n_im = 1
        elif image.ndim == 4:
            n_im, channels, rows, cols = image.shape
        else:
            print('Image must be 2D, 3D or 4D!')
            return -1

        print(f'Number of images {n_im}')

        total_patches = (rows-self.patch_size+1)*(cols-self.patch_size+1)*n_im
        if (number_training_patches > total_patches ):
            number_training_patches = total_patches
        print(f'Number of training patches {number_training_patches}')

        # number of elements in tree
        n = int((self.branching_factor**(self.number_layers+1) -
                 self.branching_factor)/(self.branching_factor-1))
        while n > (number_training_patches*n_im):
            self.number_layers -= 1
            n = int((self.branching_factor**(self.number_layers+1) -
                     self.branching_factor)/(self.branching_factor-1))
        print(f'Number of layers {self.number_layers} number of elements {n}')

        # make input
        self.tree = np.empty((n, self.patch_size*self.patch_size*channels), dtype=float) # will be overwritten

        image = np.asarray(image, order='C')
        py_km_tree_multi(image, rows, cols, channels, n_im,
                         self.patch_size, self.number_layers, self.branching_factor,
                         number_training_patches, self.normalization, self.tree)


    def search(self, image):
        '''
        Search kmtree for all patches in the image to create an assignment image.
        The assignment image is an image of indicies of closest kmtree node for
        each pixel.

        Parameters
        ----------
        image : numpy array
            2D image.

        Returns
        -------
        A : numpy array
            Assignment image of the same rows and cols as the input image.

        '''
        if self.tree is None:
            print('Tree is not build. Run build_tree first.')
            return None

        # say where to look for the function
        py_search_km_tree = lib.search_km_tree
        # say which inputs the function expects
        py_search_km_tree.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_bool,
                                 ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_search_km_tree.restype = None

        rows, cols = image.shape[0:2]
        channels = 1
        if ( image.ndim > 2 ):
            channels, rows, cols = image.shape

        number_nodes = self.tree.shape[0]
        image = np.asarray(image, order='C')

        # make input
        assignment = np.empty((rows,cols), dtype=np.int32) # will be overwritten

        py_search_km_tree(image, rows, cols, channels, self.tree, self.patch_size,
                          number_nodes, self.branching_factor, self.normalization,
                          assignment)

        return assignment


class DictionaryPropagator:
    # Class for propagating information for dictionary-based segmentatino.
    def __init__(self, dictionary_size, patch_size=15):
        '''
        Initializes a propagate object.

        Parameters
        ----------
        dictionary_size : integer
            Number of dictionary elements (nodes in the km-tree).
        patch_size : integer, optional
            Side length of the image patch. Should be an odd number.
            The default is 15.

        Returns
        -------
        None.

        '''
        self.patch_size = patch_size
        self.dictionary_size = dictionary_size
        self.probability_dictionary = None


    def improb_to_dictprob(self, assignment, probability):
        '''
        Taking image probabilities and assigning them to dictionary probabilities
        according to an assignment image A.

        Parameters
        ----------
        assignment : numpy array
            Assignment image with the same rows and cols as the input image.
        probability : numpy array
            Probability image with the same rows and cols as the image and channels
            the same as number of labels in the iamge.

        Returns
        -------
        None.

        '''


        # say where to look for the function
        py_prob_to_dict = lib.prob_im_to_dict
        # say which inputs the function expects
        py_prob_to_dict.argtypes = [ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_prob_to_dict.restype = None

        probability = np.asarray(probability, order='C')
        rows, cols = assignment.shape
        number_layers = probability.shape[0]

        # make place for input
        if (self.probability_dictionary is None) or (self.probability_dictionary.shape[1]
                                != number_layers*self.patch_size**2):
            self.probability_dictionary = np.empty((self.dictionary_size,
                                number_layers*self.patch_size**2), dtype=float) # will be overwritten

        # do the work
        py_prob_to_dict(assignment, rows, cols, probability, number_layers, self.patch_size,
                        self.dictionary_size, self.probability_dictionary)


    def dictprob_to_improb(self, assignment):

        '''
        Dictionary probabilities to iamge probabilities.

        Parameters
        ----------
        assignment : numpy array
            Assignment image.

        Returns
        -------
        probability : numpy array
            Probability image of size rows x cols x layers.

        '''
        if ( self.probability_dictionary is None ):
            print('Dictionary is not computed')
            return None

        # say where to look for the function
        py_dict_to_prob_im_opt = lib.dict_to_prob_im_opt
        # say which inputs the function expects
        py_dict_to_prob_im_opt.argtypes = [ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                  ctypes.c_int, ctypes.c_int,
                                  ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ctypes.c_int, ctypes.c_int,
                                  ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_dict_to_prob_im_opt.restype = None

        rows, cols = assignment.shape
        number_layers = int(self.probability_dictionary.shape[1]/(self.patch_size**2))


        # make input
        probability = np.empty((number_layers, rows, cols)) # will be overwritten
        py_dict_to_prob_im_opt(assignment, rows, cols, self.probability_dictionary,
                               self.patch_size, number_layers, probability)

        return probability


if __name__ == '__main__':

    # Example use
    import PIL
    import matplotlib.pyplot as plt
    import utils

    # Read image and labels.
    image = np.array(PIL.Image.open('../../data/carbon.png'))
    image = utils.normalize_to_float(image)
    labels = np.array(PIL.Image.open('../../data/carbon_labels.png'))
    labels_onehot = utils.labels_to_onehot(labels)
    nr_labels = labels.max()

    # Compute assignment image
    kmdict = KMTree(patch_size = 9,
                    branching_factor = 5,
                    number_layers = 5)
    kmdict.build(image, 30000)
    A = kmdict.search(image)

    # Make dictionary and update dictionary probabilities
    dct = DictionaryPropagator(kmdict.tree.shape[0], patch_size = 15)
    dct.improb_to_dictprob(A, labels_onehot)

    # Compute probability image and segmentation
    P = dct.dictprob_to_improb(A)
    S = utils.segment_probabilities(P)

    # Use segmentation to update dictionary probabilities once more
    dct.improb_to_dictprob(A, utils.labels_to_onehot(S))

    # Compute probability image and segmentation
    P = dct.dictprob_to_improb(A)
    S = utils.segment_probabilities(P)

    # Visualize
    fig,ax = plt.subplots(3, 2, sharex=True, sharey=True)
    ax[0,0].imshow(image, cmap='gray')
    ax[0,1].imshow(A, cmap='jet')
    ax[1,0].imshow(P[0])
    ax[1,1].imshow(P[1])
    ax[2,0].imshow(labels, vmin=0, vmax=nr_labels)
    ax[2,1].imshow(S, vmin=0, vmax=nr_labels)
    plt.show()



