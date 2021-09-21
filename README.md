## insegt(py)

*A py version of [InSegt](https://github.com/vedranaa/InSegt)*

Given lited labeling of an image, InSegt can provide full segmentation of this, or similar, image.

Input image | User labelings | Segmentation result | Screenshot
:---:|:---:|:---:|:---:
<img src="screenshots/glass/gray.png" width = "200">  |  <img src="screenshots/glass/annotations_overlay.png" width = "200"> | <img src="screenshots/glass/segmentations_overlay.png" width = "200"> | <img src="screenshots/glass/screenshot.png" width = "200">


### Instalation
* Download the code
* pip install -/path/to/insegtpy/folder/containing/setup.py/


### Use

InSegt has an interactive annotator (implemented in `insegtpy\annotators`) and, most importantly, a segmentation model. We have developed a range of segmentation models, and for some of them we provide more than one segmentation. Choosing an appropriate model may require expertise, so take a look at the provided Jupyter noteboos and python scripts to find an example that resembles your problem. 

Furthermore, a model needs to be initiated with suitable parameters. Choosing parameters often requires experiments. Here, it is a good idea to start with a small image. 

#### Python scripts

* `demos/skbasic_glasfibre_demo.py`, a demo script showing how insegt may be used with basic insegt model for interactive segmentation. This model relies on scikit-learn python package, so it may be
used if cpp code is misbihaving. As examples we use CT image of glass fibres. 

* `demos/gaussfeat_nerve_demo.py`, a demo script showing how to use multiscale segmentation based on Gaussian features clustered in KM tree. As examples we use CT nerves image. 

* `demo_insegtbasic.py`, a demo script that processes an image using functionality from `insegtbasic.py`.
   - In particular, it uses `insegtbasic.patch_clustering` function for building the dictionary and `insegtbasic.two_binarized` function for processing the label image into a segmentation image.
   - No interaction! Instead, you load an image to be segmented, and an image of the same size containing the user labeling.

<div align="center"><img src="screenshots/demo_insegtbasic.png" width = "750"></div>


* `demo_insegtbasic_processing_explained.py`, similar to  the demo above, but the processing implemented in `insegtbasic.two_binarized` is divided into steps and visualized in more detail.
  - In particular, here you have access to assignment image and the probability images for different labels.

<div align="center"><img src="screenshots/demo_insegtbasic_explained.png" width = "750"></div>

* `annotator.py`, a module containing the `Annotator` class. `Annotator`, is a widget for drawing on an image. It is based on [qt for python](https://doc.qt.io/qtforpython/). All interaction is using mouse
clicks, draws, and keyboard input. Help is accessed by pressing **H**. Built-in example uses an image from `skimage.data`.

* `insegtannotator.py`, a module containing `InSegtAnnotator` class, which is a subclass of `Annotator` extended with the functionality for interactive segmentation. To use `InsegtAnnotator` you need a processing function that given labels (annotations) returns a  segmentation.  Built-in example uses an image from `skimage.data` and a processing function based on a mean color for every label, and a pixel-to-color distance.

* `demo_km.py`, a demo showing the use of k-means clustering. Uses the module `km_dict.py`, which relies on `km_dict.cpp` compiled into `km_dict_lib.so`.

* `demo_feat.py`, a demo showing feature-based segmentation. Uses the module `feat_seg.py` which relies on `image_feat.cpp` compiled into `image_feat_lib.so`.

#### Jupyter notebooks

* `notebooks/Multiscale Gauss features InSegt on nerves.ipynp`
* `notebooks/Patch-based non-interactive fibre segmentation.ipynp`
* `notebooks/Volumetric InSegt on nerves.ipynp`

