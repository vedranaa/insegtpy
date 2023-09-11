# insegt(py)

*A py version of [InSegt](https://github.com/vedranaa/InSegt). Original matlab version was used in publications prior to 2020.*

Given lited labeling of an image, InSegt can provide full segmentation of this, or similar, image.

Input image | User labelings | Segmentation result | Screenshot
:---:|:---:|:---:|:---:
<img src="screenshots/glass/gray.png" width = "200">  |  <img src="screenshots/glass/annotations_overlay.png" width = "200"> | <img src="screenshots/glass/segmentations_overlay.png" width = "200"> | <img src="screenshots/glass/screenshot.png" width = "200">


## Installation
* `pip install git+https://github.com/vedranaa/insegtpy.git`


## Use

InSegt has an interactive annotator (implemented in `insegtpy/annotators`) and, most importantly, a segmentation model. We have developed a few different segmentation models (placed in `insegtpy/models`), and for some of them we provide more than one segmentation. For help with choosing an appropriate model and suitable parameters, take a look at the provided Jupyter notebooks and python scripts.

Models are built using two functions:
- `sk_basic_segmentor` uses intensities from image patches as features clustered using minibatch k-means from scikit-learn. Is also available in a multi-scale version.
- `gauss_features_segmentor` uses (multi-sigma) Gauss features clustered using km-tree. Also available in a multi-scale version.

#### Python demos

* `demos/skbasic_glassfibre_demo.py`, a demo script where `sk_basic_segmentor` is used for detecting fibres in a CT image of glass fibres. A good place to start, and should also run regardless of whether cpp code is misbehaving.

* `demos/gaussfeat_nerve_demo.py`, a demo script showing how to use multiscale segmentation with `gauss_features_segmentor`. As examples we use CT nerves image.

#### Jupyter notebooks

* `notebooks/Patch-based non-interactive fibre segmentation.ipynp`
* `notebooks/Multiscale Gauss features InSegt on nerves.ipynp`
* `notebooks/Volumetric InSegt on nerves.ipynp`

#### Other code and leftovers

* The main of `models/kmdict.py`, shows the use of km tree for clustering, here exemplified on patches from an image of carbon fibres. Relies on `km_dict.cpp` compiled into `km_dict_lib.so`.

* The main of `models/gaussfeat.py` shows the extraction of Gauss features from the image.* `annotators/annotator.py`, a module containing the `Annotator` class. `Annotator`, is a widget for drawing on an image. It is based on [qt for python](https://doc.qt.io/qtforpython/). All interaction is using mouse clicks, draws, and keyboard input. Help is accessed by pressing **H**.  Demo script `deomos/only_insegtannotator_demo.py` uses an image from `skimage.data`.

* `annotators/insegtannotator.py`, a module containing `InSegtAnnotator` class, which is a subclass of `Annotator` extended with the functionality for interactive segmentation. To use `InsegtAnnotator` you need a processing function that given labels (annotations) returns a  segmentation.  Demo script `deomos/only_insegtannotator_demo.py` uses an image from `skimage.data` and a processing function based on a mean color for every label, and a pixel-to-color distance.

## How InSegt works

InSegt preforms:
- Dense feature extraction. This may be intensities collected around every pixels, or other features computed in every image pixel. In fibre example below we extract patches around each pixel, so features are not extracted from boundary pixels.
- Feature clustering. This uses sing some variant of k-means clustering to cluster image features. The *assignment* image shows the outcome of pixel clustering.
- Propagating labeling. This propagates user-provided partial labelling (*labels*) according to the assignment. The outcome of propagation is a probability image for every label class (here *probability label 1* and *2*). Thresholding probability image yields *segmentation*.

<div align="center"><img src="screenshots/demo_insegtbasic_explained.png" width = "750"></div>
