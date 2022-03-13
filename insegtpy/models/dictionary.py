#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 22:51:52 2021

@author: abda
"""

# Test building a multiscale dictionary from multiple images

import matplotlib.pyplot as plt
import PIL
import numpy as np
import glob
from gaussmulti import GaussMultiImage
import insegtpy.models.utils as utils


#%%
data_dir = '/Users/abda/Documents/Projects/data/BBBC038_cells/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train/'
dir_names = glob.glob(data_dir + '*')
dir_names.sort()


images = []
labels = []
for f_name in dir_names[:128]:
    im_names = glob.glob(f_name + '/images/*.png')
    images.append(np.array(PIL.Image.open(im_names[0]).convert('L')))
    labels.append(np.array(PIL.Image.open(f_name + '/mask/mask_border.png'))//30 + 1)
#%%
fig, ax = plt.subplots(8,16)
ax = ax.ravel()
for im, axs, i in zip(images, ax, range(len(images))):
    axs.imshow(im)
    axs.set_title(f'{i}')
#%%
idx = [0,1,2,4,7,8,10,11,12,13,15,21,22,23,25,32,36,37,42,51,53,60,62,63]
fig, ax = plt.subplots(4,6)
ax = ax.ravel()
for idxs, axs in zip(idx, ax):
    axs.imshow(images[idxs])
# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for label, axs in zip(labels, ax):
#     axs.imshow(label)
#%%
ims = []
lbs = []
for idxs in idx:
    ims.append(images[idxs])
    lbs.append(labels[idxs])
images = ims
labels = lbs
#%% Final model
n_label = 0
for label in labels:
    n_label = np.maximum(n_label, label.max())

model = GaussMultiImage(scales = [1, 0.5], sigmas = [1,2,8,32], n_feat_per_image=2000,
                        branching_factor=30, number_layers=3, propagation_size=9, nr_labels = n_label)
# model = GaussMultiImage(scales = [1, .75, 0.5], sigmas = [1,4,10,20,36,52], n_feat_per_image=5000,
#                         branching_factor=14, number_layers=4, propagation_size=9, nr_labels = n_label)
model.compute_dictionary(images, labels)
prob = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)

#%%

fig, ax = plt.subplots(8,12)
ax = ax.ravel()
for im, axs in zip(images, ax):
    axs.imshow(im)

fig, ax = plt.subplots(8,12)
ax = ax.ravel()
for probability, axs in zip(prob, ax):
    axs.imshow(probability.transpose(1,2,0))

#%%

images_test = []
for f_name in dir_names[216:232]:
    im_names = glob.glob(f_name + '/images/*.png')
    images_test.append(np.array(PIL.Image.open(im_names[0]).convert('L')))



prob_test = model.compute_probability(images_test)

fig, ax = plt.subplots(4,4)
ax = ax.ravel()
for im, axs in zip(images_test, ax):
    axs.imshow(im)

fig, ax = plt.subplots(4,4)
ax = ax.ravel()
for probability, axs in zip(prob_test, ax):
    axs.imshow(probability.transpose(1,2,0))

#%%

labels_test = []
images_test = []
for f_name in dir_names[0:300]:
    im_names = glob.glob(f_name + '/images/*.png')
    images_test.append(np.array(PIL.Image.open(im_names[0]).convert('L')))
    labels_test.append(np.array(PIL.Image.open(f_name + '/mask/mask_border.png'))//30 + 1)
    
#%%
nr = 117

prob_one = model.compute_probability(images_test[nr])
seg = utils.segment_probabilities(prob_one[0])
fig, ax = plt.subplots(1, 3, sharex = True, sharey = True)
ax[0].imshow(labels_test[nr])
ax[1].imshow(seg)
ax[2].imshow(images_test[nr])
# plt.figure()
# plt.imshow(seg)
# plt.figure()
# plt.imshow(labels_test[nr])


def label_diff(seg_gt, seg_test):
    seg_gt = seg_gt-(seg_gt==3)
    seg_test = seg_test-(seg_test==3)
    return 1-np.sum(np.abs(seg_gt-seg_test)>0)/np.prod(seg_gt.shape)
    
diff_seg = label_diff(labels_test[nr],seg)

diff_zeros = label_diff(labels_test[nr],np.ones(labels_test[nr].shape))
print(f'Seg difference {diff_seg}\nSeg zeros {diff_zeros}')



seg_gt = labels_test[nr]
seg_test = seg
seg_gt = (seg_gt-(seg_gt==3)) == 2
seg_test = (seg_test-(seg_test==3)) == 2
dice = 2*(seg_gt & seg_test).sum()/(seg_gt.sum() + seg_test.sum())
print(f'dice {dice}')

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True)
ax[0][0].imshow(seg_gt)
ax[0][1].imshow(seg_test)
ax[1][0].imshow((seg_test ^ seg_gt))
ax[1][1].imshow(images_test[nr])


#%% optimize
model.optimize_dictionaries(model.assignments_list, labels, n_iter = 3, beta = 0, alpha = 0.2, noconstraint = True)


#%%
prob_list = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)

fig, ax = plt.subplots(8,8)
ax = ax.ravel()
for probability, axs in zip(prob_list, ax):
    axs.imshow(probability.transpose(1,2,0))

#%
fig, ax = plt.subplots(8,8)
ax = ax.ravel()
for im, axs in zip(images, ax):
    axs.imshow(im)

#%% change integration scale and reset
new_propatation_size = 15

for prop in model.dict_prop_sc.propagators:
    prop.patch_size = new_propatation_size
model.propagation_size = new_propatation_size
    
#%% Reset dictionary
labels_list = model.gfe_scale.scale_labels(labels)
model.dict_prop_sc.improb_to_dictprob_labels(model.assignments_list, labels_list)

#%% Compute dice
    
def get_dice(seg_gt, seg_test):
    seg_gt = (seg_gt-(seg_gt==3)) == 2
    seg_test = (seg_test-(seg_test==3)) == 2
    return 2*(seg_gt & seg_test).sum()/(seg_gt.sum() + seg_test.sum())

dice_all = []

im_prob_list = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)

for im_prob, label in zip(im_prob_list, labels):
    label_test = utils.segment_probabilities(im_prob)
    dice_all.append(get_dice(label, label_test))
print(np.array(dice_all).mean())

#%% Create test assignments

labels_tests = []
images_tests = []
for f_name in dir_names[128:228]:
    im_names = glob.glob(f_name + '/images/*.png')
    images_tests.append(np.array(PIL.Image.open(im_names[0]).convert('L')))
    labels_tests.append(np.array(PIL.Image.open(f_name + '/mask/mask_border.png'))//30 + 1)

features_list_tests = model.gfe_scale.compute_features(images_tests)
assignments_list_tests = model.km_tree_scale.search_scale(features_list_tests)

#%%
dice_all_test = []

im_prob_list_test = model.dict_prop_sc.dictprob_to_improb_scales(assignments_list_tests)

for im_prob, label in zip(im_prob_list_test, labels_tests):
    label_test = utils.segment_probabilities(im_prob)
    dice_all_test.append(get_dice(label, label_test))
print(np.array(dice_all_test).mean())


#%%


fig, ax = plt.subplots(8,12)
ax = ax.ravel()
for probability, axs, i in zip(im_prob_list_test, ax, range(len(images_tests))):
    axs.imshow(probability.transpose(1,2,0))
    axs.set_title(f'{i}')

fig, ax = plt.subplots(8,12)
ax = ax.ravel()
for im, axs in zip(images_tests, ax):
    axs.imshow(im)


#%%
a = np.array([1,2,3,4])
b = a.copy()


for c in b:
    c *= 2
    
print(a-b)

for i in range(a.shape[0]):
    b[i] *= 2
    print(i)
    
print(a-b)

#%%






# import time

# #%% Compute features

# class GaussFeatureMultiIm:
#     def __init__(self, sigmas = [1,2,4]):
#         if type(sigmas) is not list: sigmas = [sigmas]
#         self.sigmas = sigmas
#         self.normalization_means = None
#         self.normalization_stds = None

#     def compute_features(self, images):
#         feature_list = []
#         feat_sums = 0
#         feat_vars = 0
#         feat_count = 0
#         for image in images:
#             image = utils.normalize_to_float(image)
#             t = time.time()
#             # Compute features.
#             features = []
#             for sigma in self.sigmas:
#                 features.append(get_gauss_feat_im(image, sigma))
#             features = np.asarray(features).reshape((-1,)+image.shape)
#             feature_list.append(features)
#             if ( self.normalization_means is None):
#                 n = np.prod(features.shape[1:])
#                 feat_sums += np.sum(features, axis=(1,2))
#                 feat_vars += np.var(features, axis=(1,2))*n
#                 feat_count += n
#             print(time.time()-t)
        
#         if ( self.normalization_means is None):
#             self.normalization_means = feat_sums/feat_count
#             self.normalization_stds = (feat_vars/feat_count)**0.5
            
#         # Normalize
#         for features in feature_list:
#             features -= self.normalization_means.reshape((-1,1,1))
#             features *= (1/self.normalization_stds).reshape((-1,1,1))
#         return feature_list
    
    

#     def select_features(self, feature_list, n_feat_per_image = 10000):
#         feature_dim = feature_list[0].shape[0]
#         n_images = len(feature_list)
#         features_for_kmtree = np.empty((feature_dim, n_images*n_feat_per_image, 1))
        
#         t = 0
#         for features in feature_list:
#             n_feat_in_image = np.prod(features.shape[1:])
#             n_feat = np.minimum(n_feat_in_image, n_feat_per_image)
#             f = t # from index
#             t += n_feat # to index
#             random_idx = np.random.permutation(n_feat_in_image)[:n_feat]
#             features_for_kmtree[:,f:t] = features.reshape((-1,n_feat_in_image))[:,random_idx].reshape((feature_dim,-1,1))
#         return features_for_kmtree


# class GaussFeatureScale:
#     def __init__(self, scales = [1, 0.5, 0.25], sigmas = [1, 2, 4], n_feat_per_image = 10000):
#         self.scales = scales
#         self.sigmas = sigmas
#         self.n_feat_per_image = n_feat_per_image
#         self.gauss_feat_extractor = GaussFeatureMultiIm(sigmas)
    
#     def compute_features(self, images):
#         # images_scaled = []
#         features_list = []
#         for s in self.scales:
#             image_list_scaled = [utils.imscale(im, scale=s) for im in images]
#             # images_scaled.append(image_list_scaled)
#             feature_list_scaled = self.gauss_feat_extractor.compute_features(image_list_scaled)
#             features_list.append(feature_list_scaled)
#         return features_list
#         # return images_scaled, features_list

#     def get_features_kmtree(self, features):
#         features_for_kmtree_list = []
#         for feat in features:
#             features_for_kmtree_list.append(
#                 self.gauss_feat_extractor.select_features(feat, self.n_feat_per_image))
#         return features_for_kmtree_list
        
#     def scale_labels(self, labels):
#         labels_list = []
#         for s in self.scales:
#             labels_list.append([utils.imscale(l, scale = s, interpolation='nearest') for l in labels])
#         return labels_list
            

# class KMTreeScale:
#     def __init__(self, patch_size = 1, branching_factor = 5, number_layers = 5, 
#                  normalization = False):
#         self.patch_size = patch_size
#         self.branching_factor = branching_factor
#         self.number_layers = number_layers
#         self.normalization = normalization
#         self.km_trees = []
        
        
#     def build_scale(self, features_for_kmtree):
#         for feat_km in features_for_kmtree:
#             km_tree = KMTree(patch_size = self.patch_size, branching_factor = self.branching_factor, 
#                               number_layers = self.number_layers, normalization = self.normalization)
#             km_tree.build(feat_km, number_training_patches=feat_km.shape[1])
#             self.km_trees.append(km_tree)
            
#     def search_scale(self, features_list):
#         assignments_list = []
#         for feat, km_tree in zip(features_list, self.km_trees):
#             assignments = [km_tree.search(f) for f in feat]
#             assignments_list.append(assignments)
#         return assignments_list
    
        
# class DictionaryPropagatorScale:
#     def __init__(self, dictionary_sizes, patch_size=15):
#         self.patch_size = patch_size
#         self.dictionary_sizes = dictionary_sizes
#         self.propagators = [DictionaryPropagator(ds, patch_size) for ds in dictionary_sizes]
    
    
#     def improb_to_dictprob_scales(self, assignments_list, labels_list):
#         for assignments, labels, propagator in zip(assignments_list, labels_list, self.propagators):
#             n_images = len(assignments)
#             labels_onehot = utils.labels_to_onehot(labels[0])
#             propagator.improb_to_dictprob(assignments[0], labels_onehot)
#             probability_dictionary = propagator.probability_dictionary.copy()
#             for label, assignment in zip(labels[1:], assignments[1:]):
#                 labels_onehot = utils.labels_to_onehot(label)
#                 propagator.improb_to_dictprob(assignment, labels_onehot)
#                 probability_dictionary += propagator.probability_dictionary
#             propagator.probability_dictionary = probability_dictionary/float(n_images)
    
    
#     def dictprob_to_improb_scales(self, assignments_list):
#         probability_list = []
#         for assignment in assignments_list[0]:
#             probability_list.append(self.propagators[0].dictprob_to_improb(assignment))
#         if ( len(assignments_list) > 1 ):
#             for assignment_list, propagator in zip(assignments_list[1:], self.propagators[1:]):
#                 for assignment, probability in zip(assignment_list, probability_list):
#                     probability *= utils.imscale(propagator.dictprob_to_improb(assignment), size = probability.shape[1:])
#         for probability in probability_list:
#             p_sum = np.sum(probability, axis=0, keepdims = True)
#             p_sum = p_sum + ( p_sum == 0 )
#             probability /= p_sum
#         return probability_list



# class GaussMultiImage:
#     def __init__(self, scales = [1, 0.5, 0.25], 
#                        sigmas = [1, 3, 8],
#                        n_feat_per_image = 10000,
#                        branching_factor = 5, 
#                        number_layers = 5,
#                        propagation_size = 9,
#                        ):
#         self.scales = scales
#         self.sigmas = sigmas
#         self.n_feat_per_image = n_feat_per_image
#         self.branching_factor = branching_factor
#         self.number_layers = number_layers
#         self.propagation_size = propagation_size
#         self.km_tree_scale = None
#         self.dict_prop_sc = None
#         self.gfe_scale = None
    
#     def compute_dictionary(self, images, labels, return_probability = True):
#         self.gfe_scale = GaussFeatureScale(scales = self.scales, sigmas = self.sigmas, 
#                                       n_feat_per_image = self.n_feat_per_image)
#         features_list = self.gfe_scale.compute_features(images)
#         features_for_kmtree_list = self.gfe_scale.get_features_kmtree(features_list)
#         labels_list = self.gfe_scale.scale_labels(labels)
    
#         self.km_tree_scale = KMTreeScale(patch_size = 1, branching_factor = self.branching_factor, 
#                                     number_layers = self.number_layers)
#         self.km_tree_scale.build_scale(features_for_kmtree_list)
#         assignments_list = self.km_tree_scale.search_scale(features_list)
        
#         self.dict_prop_sc = DictionaryPropagatorScale(dictionary_sizes = 
#                                                       [kmt.tree.shape[0] for kmt in self.km_tree_scale.km_trees], 
#                                                       patch_size=self.propagation_size)
#         self.dict_prop_sc.improb_to_dictprob_scales(assignments_list, labels_list)
#         if return_probability:
#             return self.dict_prop_sc.dictprob_to_improb_scales(assignments_list)
        
#     def compute_probability(self, images):
#         if ( self.dict_prop_sc is not None ):
#             if type(images) is not list: images = [images]
#             features = self.gfe_scale.compute_features(images)
#             assignments = self.km_tree_scale.search_scale(features)
#             return self.dict_prop_sc.dictprob_to_improb_scales(assignments)
        

# #%% Multiple scales

# gfe_scale = GaussFeatureScale(scales = [1, 0.8, 0.6], sigmas = [1,5,16], n_feat_per_image = 10000)
# images_scaled, features_list, features_for_kmtree_list = gfe_scale.compute_features(images)
# labels_list = gfe_scale.scale_labels(labels)

# km_tree_scale = KMTreeScale(patch_size = 1, branching_factor = 15, number_layers = 4)
# km_tree_scale.build_scale(features_for_kmtree_list)
# assignments_list = km_tree_scale.search_scale(features_list)

# #%%

# dict_prop_sc = DictionaryPropagatorScale(dictionary_sizes = [kmt.tree.shape[0] for kmt in km_tree_scale.km_trees], patch_size=9)
# dict_prop_sc.improb_to_dictprob_scales(assignments_list, labels_list)
# probability_list = dict_prop_sc.dictprob_to_improb_scales(assignments_list)


# #%%



# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for probability, axs in zip(probability_list, ax):
#     axs.imshow(probability.transpose(1,2,0))


# #%%
# images_test = []
# for f_name in dir_names[500:516]:
#     im_names = glob.glob(f_name + '/images/*.png')
#     images_test.append(np.array(PIL.Image.open(im_names[0]).convert('L')))

# features_list_test = gfe_scale.compute_features(images_test)[1]
# assignments_list_test = km_tree_scale.search_scale(features_list_test)
# probability_list_test = dict_prop_sc.dictprob_to_improb_scales(assignments_list_test)


# #%%

# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for probability, axs in zip(probability_list_test, ax):
#     axs.imshow(probability.transpose(1,2,0))

# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for im, axs in zip(images_test, ax):
#     axs.imshow(im)



# #%% Single scale

# gauss_feat_extractor = GaussFeatureMultiIm(sigmas = [1,2,5,10])

# feature_list = gauss_feat_extractor.compute_features(images)

# features_for_kmtree = gauss_feat_extractor.select_features(feature_list, n_feat_per_image = 10000)


# #%%

# km_tree = KMTree(patch_size = 1, branching_factor = 30, number_layers = 3)
# km_tree.build(features_for_kmtree, number_training_patches=features_for_kmtree.shape[1])

# assignment_list = []
# for features in feature_list:
#     assignment_list.append(km_tree.search(features))

# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for assignment, axs in zip(assignment_list, ax):
#     axs.imshow(assignment)

# #%%

# n_images = len(images)

# propagator = DictionaryPropagator(dictionary_size=km_tree.tree.shape[0], patch_size=9)

# labels_onehot = utils.labels_to_onehot(labels[0])
# propagator.improb_to_dictprob(assignment_list[0], labels_onehot)
# probability_dictionary = propagator.probability_dictionary.copy()

# tmp_dict_prob = []
# tmp_dict_prob.append(propagator.probability_dictionary.copy())

# for label, assignment in zip(labels[1:], assignment_list[1:]):
#     labels_onehot = utils.labels_to_onehot(label)
#     propagator.improb_to_dictprob(assignment, labels_onehot)
#     probability_dictionary += propagator.probability_dictionary
#     tmp_dict_prob.append(propagator.probability_dictionary.copy())
# propagator.probability_dictionary = probability_dictionary/float(n_images)

# probability_list = []
# for assignment in assignment_list:
#     probability_list.append(propagator.dictprob_to_improb(assignment))

# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for probability, axs in zip(probability_list, ax):
#     axs.imshow(probability.transpose(1,2,0))

# #%%

# plt.figure()
# plt.imshow(tmp_dict_prob[0]-tmp_dict_prob[2])
#         # self.segt_list = [model_init(utils.imscale(image, s, 'linear')) for s in scales]
# #%%


# images_test = []
# for f_name in dir_names[:16]:
#     im_names = glob.glob(f_name + '/images/*.png')
#     images_test.append(np.array(PIL.Image.open(im_names[0]).convert('L')))

# feature_list_test = gauss_feat_extractor.compute_features(images_test)

# assignment_list_test = []
# for features in feature_list_test:
#     assignment_list_test.append(km_tree.search(features))

# probability_list_test = []
# for assignment in assignment_list_test:
#     probability_list_test.append(propagator.dictprob_to_improb(assignment))

# fig, ax = plt.subplots(4,4)
# ax = ax.ravel()
# for probability, axs in zip(probability_list_test, ax):
#     axs.imshow(probability.transpose(1,2,0))




