#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Basic InSegt functionality.

This module provides basic InSegt image processing functionallity. It uses
intensities from image patches as features for clustering. For clustering it
uses minibatch k-means from sklearn.


Author: vand@dtu.dk, 2020

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.cluster
import scipy.sparse
import insegtpy.models.segt as segt

# HELPING FUNCTIONS
# (originaly written for use without annotator)

def image2patches(image, patch_size, stepsize=1):
    """Rearrange image patches into columns
    Arguments:
        image: a 2D image, shape (X,Y).
        patch size: size of extracted squared paches.
        stepsize: patch step size.
    Returns:
        patches: a 2D array which in every column has a patch associated with
            one image pixel. For stepsize 1, the number of returned patches
            is (X-patch_size+1)*(Y-patch_size+1) due to bounary. The length
            of columns is patch_size**2.
    """
    X, Y = image.shape
    s0, s1 = image.strides
    nrows = X-patch_size+1
    ncols = Y-patch_size+1
    shp = patch_size, patch_size, nrows, ncols
    strd = s0, s1, s0, s1
    out_view = np.lib.stride_tricks.as_strided(image, shape=shp, strides=strd)
    return out_view.reshape(patch_size*patch_size,-1)[:,::stepsize]


def ndimage2patches(im, patch_size, stepsize=1):
    """Rearrange image patches into columns for N-D image (e.g. RGB image)."""""
    if im.ndim == 2:
        return image2patches(im, patch_size, stepsize)
    elif im.ndim == 3:
        X ,Y, L = im.shape
        patches = np.zeros((L*patch_size*patch_size,
                            (X - patch_size + 1)*(Y - patch_size + 1)))
        for i in range(L):
            patches[i*patch_size**2:(i+1)*patch_size**2,:] = image2patches(
                im[:,:,i], patch_size, stepsize)
        return patches
    elif im.ndim == 4:
        N, X, Y, L = im.shape
        patches_per_im = (X - patch_size + 1)*(Y - patch_size + 1)
        patches = np.empty((L*patch_size*patch_size,
                            N*patches_per_im))
        for i in range(L):
            for n in range(N):
                patches[i*patch_size**2:(i+1)*patch_size**2,n*patches_per_im:(n+1)*patches_per_im] = image2patches(
                    im[n,:,:,i], patch_size, stepsize
                )
        return patches
    else:
        raise ValueError('Image dimension not supported.')


def image2assignment(image, patch_size, nr_clusters, nr_training_patches):
    """ Extract, cluster and assign image patches using minibatch k-means."""
    patches = ndimage2patches(image, patch_size)
    patches_subset = patches[:,np.random.permutation(np.arange(patches.shape[1]))
                             [:nr_training_patches]]
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters = nr_clusters,
            max_iter = 10, batch_size = 3*nr_clusters)
    kmeans.fit(patches_subset.T)
    assignment = kmeans.predict(patches.T)
    if image.ndim < 4:
        assignment = assignment.reshape((image.shape[0] - patch_size + 1,
                                         image.shape[1] - patch_size + 1))
    else:
        assignment = assignment.reshape((image.shape[0],
                                         image.shape[1] - patch_size + 1,
                                         image.shape[2] - patch_size + 1))
    return assignment, kmeans



def new2assignment(image, kmeans):
    """ Extract and assign image patches using pre-clustered k-means."""
    patch_size = int(np.sqrt(kmeans.cluster_centers_.shape[1]))
    patches = ndimage2patches(image, patch_size)
    assignment = kmeans.predict(patches.T)
    return assignment.reshape((image.shape[0] - patch_size + 1,
                               image.shape[1] - patch_size + 1))


def assignment2biadjacency(assignment, image_shape, patch_size, nr_clusters):
    """ Algorithm 1 from https://arxiv.org/pdf/1809.02226.pdf"""
    # Ensure first dim is number of images
    if len(image_shape) < 4:
        image_shape = (1,) + image_shape
    if assignment.ndim < 3:
        assignment = assignment[None]
    n = image_shape[0]*image_shape[1]*image_shape[2]
    m = patch_size*patch_size*nr_clusters
    s = (patch_size-1)//2
    # find displacement in i and j for within-patch positions dx and dy.
    dy, dx = np.meshgrid(np.arange(patch_size)-s,np.arange(patch_size)-s)
    di = (dy + image_shape[2]*dx).ravel()
    dj = (dy + patch_size*dx).ravel()
    # populate index list for every assignment.
    i_accumulate = np.empty((assignment.size,patch_size**2))
    j_accumulate = np.empty((assignment.size,patch_size**2))
    for b in range(assignment.shape[0]):
        for x in range(assignment.shape[1]):
            for y in range(assignment.shape[2]):
                k = assignment[b,x,y]
                i = (y+s) + (x+s)*image_shape[2] + b*image_shape[1]*image_shape[2] # linear index of the image pixel
                j = s + s*patch_size + k*patch_size**2 # linear index of the patch center
                p = y + x*assignment.shape[2] + b*assignment.shape[1]*assignment.shape[2]
                i_accumulate[p] = i+di
                j_accumulate[p] = j+dj
    B = scipy.sparse.coo_matrix((np.ones(i_accumulate.size, dtype=bool),
                    (i_accumulate.ravel(),j_accumulate.ravel())),shape=(n,m))
    return B.tocsr()


def biadjacency2transformations(B):
    """ Eq. (6) and Eq. (7) from https://arxiv.org/pdf/1809.02226.pdf"""
    s1 = np.asarray(B.sum(axis=0)) # length m
    s2 = np.asarray(B.sum(axis=1)) # length n
    s1[s1==0] = 1 # preventing division by zero
    s2[s2==0] = 1
    s1 = 1/s1
    s2 = 1/s2
    T1 = scipy.sparse.diags(s1.ravel())*B.transpose()
    T2 = scipy.sparse.diags(s2.ravel())*B
    return (T1, T2)


def labels2labcol(labels, nr_classes):
    """ Unfold labeling image into columns for matrix multiplication."""
    labels = np.ravel(labels)
    labeled = labels>0
    labcol = scipy.sparse.coo_matrix((np.ones(np.sum(labeled), dtype=bool),
                    (np.where(labeled)[0], labels[labeled]-1)),
                    shape=(labels.size, nr_classes)).tocsr()
    return labcol


def probcol2probabilities(probcol, image_shape):
    """ Fold columns of probabilities into probability image."""
    p = np.sum(probcol, axis=1)
    nonempty = p>0
    probcol[nonempty] = probcol[nonempty]/(p[nonempty].reshape((-1,1)))
    return probcol.reshape(image_shape + (-1,))


def probcol2labcol(probcol):
    """ Probability to labels using max approach."""
    p = np.sum(probcol, axis=1)
    nonempty = p>0
    nr_nonempty = np.sum(nonempty)
    l = np.empty((nr_nonempty,), dtype=np.uint8)  # max 255 labels
    if nr_nonempty > 0: # argmax can't handle empty labeling
        np.argmax(probcol[nonempty], axis=1, out=l)
    labcol = scipy.sparse.coo_matrix((np.ones(np.sum(nonempty), dtype=bool),
                    (np.where(nonempty)[0], l)),
                    shape=probcol.shape).tocsr()
    return labcol


def gray_cool(nr_classes):
    """ Colormap as in original InSegt """
    colors = plt.cm.cool(np.linspace(0, 1, nr_classes))
    colors = np.r_[np.array([[0.5, 0.5, 0.5, 1]]), colors]
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap


def patch_clustering(image, patch_size, nr_training_patches, nr_clusters):
    """"InSegt preprocessing function: clustering, assignment and transformations."""
    print('assign')
    assignment, kmeans = image2assignment(image, patch_size,
            nr_clusters, nr_training_patches)
    print('B')
    B = assignment2biadjacency(assignment, image.shape, patch_size, nr_clusters)
    print('T1 and T2')
    T1, T2 = biadjacency2transformations(B)
    return T1, T2, kmeans


def two_binarized(labels, T1, T2):
    """InSegt processing function: from labels to probabilities."""
    nr_classes = np.max(labels)
    labcol = labels2labcol(labels, nr_classes=nr_classes) # columns with binary labels
    probcol = T2 * ((T1) * labcol) # first linear diffusion
    probcol = np.asarray(probcol.todense()) # columns with probabilities
    labcol = probcol2labcol(probcol) # binarizing labels
    dict_labels = T1 * labcol
    probcol = T2*dict_labels # second linear diffusion
    probcol = np.asarray(probcol.todense())
    probabilities = np.transpose(probcol.reshape(labels.shape + (-1,)), (2,0,1))
    return probabilities, dict_labels


def single_update(labels, T1, T2, nr_classes=None):
    """InSegt processing function: from labels to probabilities."""
    if nr_classes is None: nr_classes = np.max(labels)
    labcol = labels2labcol(labels, nr_classes=nr_classes) # columns with binary labels
    dict_labels = T1 * labcol
    probcol = T2*dict_labels # second linear diffusion
    probcol = np.asarray(probcol.todense())
    probabilities = np.moveaxis(probcol.reshape(labels.shape + (-1,)), -1, 0)
    return probabilities, dict_labels


def new2probs(image, kmeans, dict_labels):
    this_assignment = new2assignment(image, kmeans)

    patch_size = int(np.sqrt(kmeans.cluster_centers_.shape[1]))
    nr_clusters = kmeans.cluster_centers_.shape[0]

    B = assignment2biadjacency(this_assignment, image.shape, patch_size, nr_clusters)
    T2 = biadjacency2transformations(B)[1]
    probcol = T2 * dict_labels
    probcol = np.asarray(probcol.todense())
    return np.transpose(probcol.reshape(image.shape[:2] + (-1,)), (2,0,1))


# MAKING ALL OF THIS WORK WITH INSEGT
class SkBasic(segt.Segt):
    ''' A class for basic processing, to comply with insegt.
    '''

    def __init__(self, image, patch_size, nr_training_patches, nr_clusters):

        T1, T2, kmeans = patch_clustering(image, patch_size,
                        nr_training_patches, nr_clusters)
        self.T1 = T1
        self.T2 = T2
        self.kmeans = kmeans

        self.dict_labels = None

    def process(self, labels, nr_classes=None):
        probabilities, dict_labels = single_update(labels, self.T1, self.T2, nr_classes)
        self.dict_labels = dict_labels
        return probabilities

    def segment_new(self, image):

        return new2probs(image, self.kmeans, self.dict_labels)


def sk_basic_segmentor(image, patch_size=3, nr_training_patches=1000, nr_clusters=100,
             scales=None, propagation_repetitions=None):
    ''' Convenience function to create a model based on SkBasic. '''

    if scales is None or scales==1 or scales==[1]:
        print('Bulding single-scale SkBasic model.')
        model = SkBasic(image, patch_size, nr_training_patches, nr_clusters)
    else:
        print('Bulding multi-scale SkBasic model.')
        if type(scales) is not list: scales = [scales]

        model_init = lambda im: SkBasic(im, patch_size,
                                        nr_training_patches, nr_clusters)
        model = segt.Multiscale(image, scales, model_init)

    if propagation_repetitions and propagation_repetitions>1:
        print('Adding propagation repetitions.')
        model = segt.Repeated(model, propagation_repetitions)

    return model



if __name__ == '__main__':

# Example use without annotator (and without Segt)

    import PIL
    import utils

    # Read image and labels.
    image = np.array(PIL.Image.open('../../data/glass.png'))
    labels = np.array(PIL.Image.open('../../data/glass_labels.png'))
    nr_classes = np.max(labels)

    # Chooose settings.
    patch_size = 9
    nr_training_patches = 10000 # number of randomly extracted patches for clustering
    nr_clusters = 1000 # number of dictionary clusters

    # Pre-process.
    T1, T2 = patch_clustering(image, patch_size, nr_training_patches, nr_clusters)[:2]

    # Process.
    probabilities = two_binarized(labels, T1, T2)[0]
    segmentation  = utils.segment_probabilities(probabilities)

    # Visualize.
    fig, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(15,5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('image')
    ax[1].imshow(labels, vmin=0, vmax=nr_classes, cmap = gray_cool(nr_classes))
    ax[1].set_title('labels')
    ax[2].imshow(segmentation, vmin=0, vmax=nr_classes, cmap = gray_cool(nr_classes))
    ax[2].set_title('segmentation')