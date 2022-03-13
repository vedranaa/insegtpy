#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:40:09 2021

@author: abda
"""

import matplotlib.pyplot as plt
import PIL
import numpy as np
import glob
from gaussmulti import GaussMultiImage
import insegtpy.models.utils as utils


#%%
data_dir = '/Users/abda/Documents/Projects/data/EM_ISBI_Challenge/images/'
im_names = glob.glob(data_dir + 'train_images/*.png')
im_names.sort()
label_names = glob.glob(data_dir + 'train_labels/*.png')
label_names.sort()

im_sc = 1

images = []
labels = []
n = 27
# for im_name, label_name in zip(im_names[:n], label_names[:n]):
#     images.append(utils.imscale(np.array(PIL.Image.open(im_name)), scale = im_sc))
#     labels.append(utils.imscale(np.array(PIL.Image.open(label_name))//255 + 1, scale = im_sc, interpolation='nearest'))
for im_name, label_name in zip(im_names[:n], label_names[:n]):
    images.append(utils.imscale(np.array(PIL.Image.open(im_name)), scale = im_sc))
    labels.append(utils.imscale(np.array(PIL.Image.open(label_name))//255 + 1, scale = im_sc, interpolation='nearest'))
    images.append(utils.imscale(np.array(PIL.Image.open(im_name).rotate(90)), scale = im_sc))
    labels.append(utils.imscale(np.array(PIL.Image.open(label_name).rotate(90))//255 + 1, scale = im_sc, interpolation='nearest'))
    images.append(utils.imscale(np.array(PIL.Image.open(im_name).rotate(180)), scale = im_sc))
    labels.append(utils.imscale(np.array(PIL.Image.open(label_name).rotate(180))//255 + 1, scale = im_sc, interpolation='nearest'))
    images.append(utils.imscale(np.array(PIL.Image.open(im_name).rotate(270)), scale = im_sc))
    labels.append(utils.imscale(np.array(PIL.Image.open(label_name).rotate(270))//255 + 1, scale = im_sc, interpolation='nearest'))
#     images.append(utils.imscale(np.array(PIL.Image.open(im_name)).T, scale = im_sc))
#     labels.append(utils.imscale(np.array(PIL.Image.open(label_name)).T//255 + 1, scale = im_sc, interpolation='nearest'))
#     images.append(utils.imscale(np.array(PIL.Image.open(im_name).rotate(90)).T, scale = im_sc))
#     labels.append(utils.imscale(np.array(PIL.Image.open(label_name).rotate(90)).T//255 + 1, scale = im_sc, interpolation='nearest'))
#     images.append(utils.imscale(np.array(PIL.Image.open(im_name).rotate(180)).T, scale = im_sc))
#     labels.append(utils.imscale(np.array(PIL.Image.open(label_name).rotate(180)).T//255 + 1, scale = im_sc, interpolation='nearest'))
#     images.append(utils.imscale(np.array(PIL.Image.open(im_name).rotate(270)).T, scale = im_sc))
#     labels.append(utils.imscale(np.array(PIL.Image.open(label_name).rotate(270)).T//255 + 1, scale = im_sc, interpolation='nearest'))

#%%

fig, ax = plt.subplots(3,9)
ax = ax.ravel()
for im, axs, i in zip(images, ax, range(len(images))):
    axs.imshow(im)
    axs.set_title(f'{i}')

fig, ax = plt.subplots(3,9)
ax = ax.ravel()
for label, axs, i in zip(labels, ax, range(len(images))):
    axs.imshow(label)
    axs.set_title(f'{i}')


#%% Final model

model = GaussMultiImage(scales=[1, 0.75, 0.5], sigmas=[1, 2, 4, 8, 16], n_feat_per_image=2500,
                        branching_factor=30, number_layers=3, propagation_size=9)
prob = model.compute_dictionary(images, labels)

#%% optimize
model.optimize_dictionaries(model.assignments_list, labels, n_iter=1, beta=0, alpha=0.2)
# prob_list = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)

#%% Compute dice

def get_dice(seg_gt, seg_test):
    seg_gt = seg_gt == 1
    seg_test = seg_test == 1
    return 2*(seg_gt & seg_test).sum()/(seg_gt.sum() + seg_test.sum())

dice_all = []

im_prob_list = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)

for im_prob, label in zip(im_prob_list, labels):
    label_test = utils.segment_probabilities(im_prob)
    dice_all.append(get_dice(label, label_test))
print(np.array(dice_all).mean())


#%%

prob_list = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)
fig, ax = plt.subplots(3,9)
ax = ax.ravel()
for label, axs, i in zip(labels, ax, range(len(images))):
    axs.imshow(label)
    axs.set_title(f'{i}')
    
# ax = ax.ravel()
# for prob, axs, i in zip(prob_list, ax, range(len(images))):
#     axs.imshow(utils.segment_probabilities(prob))
#     axs.set_title(f'{i}')

fig, ax = plt.subplots(3,9)
ax = ax.ravel()
for prob, axs, i in zip(im_prob_list, ax, range(len(images))):
    prob[prob<0] = 0
    prob[prob>1] = 1
    axs.imshow(prob[1])
    # axs.imshow(utils.segment_probabilities(prob))
    axs.set_title(f'{i}')
#%% Create test assignments

labels_tests = []
images_tests = []
for im_name, label_name in zip(im_names[n:], label_names[n:]):
    images_tests.append(utils.imscale(np.array(PIL.Image.open(im_name)), scale = im_sc))
    labels_tests.append(utils.imscale(np.array(PIL.Image.open(label_name))//255 + 1, scale = im_sc, interpolation='nearest'))

features_list_tests = model.gfe_scale.compute_features(images_tests)
assignments_list_tests = model.km_tree_scale.search_scale(features_list_tests)

#%%
dice_all_test = []

im_prob_list_test = model.dict_prop_sc.dictprob_to_improb_scales(assignments_list_tests)

for im_prob, label in zip(im_prob_list_test, labels_tests):
    # im_prob[0] *= 1.5
    label_test = utils.segment_probabilities(im_prob)
    dice_all_test.append(get_dice(label, label_test))
print(np.array(dice_all_test).mean())


#%%

fig, ax = plt.subplots(3,3,sharex=True,sharey=True)
for i in range(len(images_tests)):
    im_prob_list_test[i][im_prob_list_test[i]<0] = 0
    im_prob_list_test[i][im_prob_list_test[i]>1] = 1
    ax[0][i].imshow(images_tests[i])
    ax[1][i].imshow(im_prob_list_test[i][0])
    ax[2][i].imshow(labels_tests[i])



#%%

fig, ax = plt.subplots(1,3)
ax = ax.ravel()
for im, axs, i in zip(images_tests, ax, range(len(images))):
    axs.imshow(im)
    axs.set_title(f'{i}')

fig, ax = plt.subplots(1,3)
ax = ax.ravel()
for prob, axs, i in zip(im_prob_list_test, ax, range(len(images))):
    axs.imshow(prob[0]<0.5)
    axs.set_title(f'{i}')

fig, ax = plt.subplots(1,3)
ax = ax.ravel()
for label, axs, i in zip(labels_tests, ax, range(len(images))):
    axs.imshow(label)
    axs.set_title(f'{i}')



#%% change integration scale and reset
new_propatation_size = 15

for prop in model.dict_prop_sc.propagators:
    prop.patch_size = new_propatation_size
model.propagation_size = new_propatation_size

#%% Reset dictionary
labels_list = model.gfe_scale.scale_labels(labels)
model.dict_prop_sc.improb_to_dictprob_labels(model.assignments_list, labels_list)


#%%
fig, ax = plt.subplots(6,10)
ax = ax.ravel()
for axs, i in zip(ax, range(60)):
    j = int(np.floor(i/4) + (i%4)*15)
    print(j)
    axs.imshow(model.features_list[0][10][j])
    axs.set_title(f'{i}')

#%% Try non-annotated images
data_dir = '/Users/abda/Documents/Projects/data/EM_ISBI_Challenge/images/'
imt_names = glob.glob(data_dir + 'test_images/*.png')
imt_names.sort()


#%% Create test assignments


n_im = 30

images_no_ann = []
for im_name in imt_names:
    images_no_ann.append(utils.imscale(np.array(PIL.Image.open(im_name)), scale = im_sc))
    
features_list_no_ann = model.gfe_scale.compute_features(images_no_ann[:n_im])
assignments_list_no_ann = model.km_tree_scale.search_scale(features_list_no_ann)

#%% Show

im_prob_list_no_ann = model.dict_prop_sc.dictprob_to_improb_scales(assignments_list_no_ann)


fig, ax = plt.subplots(2,n_im,sharex=True,sharey=True)
for i in range(len(im_prob_list_no_ann)):
    im_prob_list_no_ann[i][im_prob_list_no_ann[i]<0] = 0
    im_prob_list_no_ann[i][im_prob_list_no_ann[i]>1] = 1
    ax[0][i].imshow(images_no_ann[i])
    ax[1][i].imshow(im_prob_list_no_ann[i][0])


#%%

out_dir = '/Users/abda/Documents/Projects/Code/TextureGUI/results/isbi/'

for i, im in enumerate(im_prob_list_no_ann):
    im_out = PIL.Image.fromarray((255*im[0]).astype(np.uint8))
    im_out.save(out_dir + f'test_result_{i:02d}.png', 'PNG')


#%% Peripheral nerves


import matplotlib.pyplot as plt
import PIL
import numpy as np
import glob
from gaussmulti import GaussMultiImage
import insegtpy.models.utils as utils


#%%


im_sc = 1


npz_filename = '/Users/abda/Documents/Projects/data/nerve_data/data_central_slices.npz'

npzfile = np.load(npz_filename)
images = npzfile['arr_0']
labels = npzfile['arr_1']
mask = npzfile['arr_2'] 

#%%
import skimage.transform
image = skimage.transform.resize(images[9], (1024,1024), preserve_range=True)
label = skimage.transform.resize(labels[9], (1024,1024), preserve_range=True, order=0, anti_aliasing=False)
# image = images[9]
# label = labels[9]
label += label==0
label = label.astype(np.uint8)
fig,ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[1].imshow(label) 

#%% Final model

model = GaussMultiImage(scales=[1, 0.5, 0.25], sigmas=[1, 2, 4, 8], n_feat_per_image=100000,
                        branching_factor=30, number_layers=3, propagation_size=9)
prob = model.compute_dictionary([image], [label.astype(np.uint8)])

#%% optimize
model.optimize_dictionaries(model.assignments_list, [label], n_iter=5, beta=0, alpha=0.5, noconstraint=True)

#%%
fig, ax = plt.subplots(1)
ax.imshow(prob[0].transpose(1,2,0))

#%%

prob_list = model.dict_prop_sc.dictprob_to_improb_scales(model.assignments_list)
fig, ax = plt.subplots(1)
ax.imshow(prob_list[0].transpose(1,2,0))

#%%
im_name = '/Users/abda/Documents/Projects/MAX4Imagers/data/NT2_png/NT2_0001.png'
im_test = np.array(PIL.Image.open(im_name)).astype(np.float)
im_test = skimage.transform.resize(im_test, image.shape, preserve_range=True)

features_list_test = model.gfe_scale.compute_features([im_test])
assignments_list_test = model.km_tree_scale.search_scale(features_list_test)
              
#%%

prob_test = model.dict_prop_sc.dictprob_to_improb_scales(assignments_list_test)
fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].imshow(im_test)
# ax[1].imshow(prob_test[0][0])
ax[1].imshow(prob_test[0].transpose(1,2,0))




#%% change integration scale and reset
new_propatation_size = 9

for prop in model.dict_prop_sc.propagators:
    prop.patch_size = new_propatation_size
model.propagation_size = new_propatation_size

#%% Reset dictionary
labels_list = model.gfe_scale.scale_labels([label])
model.dict_prop_sc.improb_to_dictprob_labels(model.assignments_list, labels_list)























