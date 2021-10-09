import numpy as np
#import skimage.io
import cv2


def normalize_to_float(image):
    '''
    Set the image values to float and in the range of 0-1, if the
    input image is uint8, uint16, or uint32.

    Parameters
    ----------
    image : numpy array
        Input image.

    Returns
    -------
    image : numpy array
        Output image.

    '''
    if image.dtype == 'uint8':
        image = image.astype(np.float)/255.0
    elif image.dtype == 'uint16':
        image = image.astype(np.float)/65535.0
    elif image.dtype == 'uint32':
        image = image.astype(np.float)/4294967295.0
    return image

# def read_label_image(filename):
#     ''' 
#     Reads label image saved as rgba image using skimage.
#     Alternatively use labels = np.array(PIL.Image.open(filename))
    
#     Takes a filename of an rgb(a) image with pixels values representing N+1 
#     distinct colors. Returns a 2D array with values from 0 to N. Value 0 is
#     given to dominant color, value 1 to second-dominant etc.
#     '''
    
#     labels = skimage.io.imread(filename)
#     u = np.unique(labels.reshape((-1, labels.shape[-1])), axis=0, 
#                   return_inverse=True, return_counts=True)
#     s = np.argsort(u[2])[::-1]
#     labels = s[u[1]].reshape(labels.shape[:2]) # (r,c) with elements 0,1,2...
#     return labels
    

def labels_to_onehot(labels, nr_labels = None):
    ''' 
    From (r,c) image containing values 0 to N into onehot labels (r,c,N+1).
    N is either computed as the largest value in labels or given by nr_labels.
    Onehot labels have all zeros in unlabeled pixels, i.e. pixels that had
    value 0. 
    
    '''
    if nr_labels is None: nr_labels = labels.max()
    
    lmat = np.concatenate((np.zeros((1,nr_labels)), np.eye(nr_labels)))
    labels = lmat[labels].transpose((2, 0, 1))
    return labels
    
def segment_probabilities(probabilities):
    '''
    Probabilities to segmentation using max-prob approach. Pixels with
    all labels probabilities being 0 will get value 0.
    '''
    segmentation = np.zeros(probabilities.shape[1:], dtype=np.uint8)  # max 255 labels
    if probabilities.shape[0]>1:
        uncertain = probabilities.sum(axis=0)==0
        # TODO concider adding uncertain = (probabilities == probabilities[0]).all(axis=0)
        np.argmax(probabilities, axis=0, out=segmentation)
        segmentation += 1
        segmentation[uncertain] = 0
    elif probabilities.shape[0]==1:
        segmentation[probabilities[0]>0] = 1
    return segmentation
    

def normalize_to_one(probabilities):
    '''
    Normalize labels probabilities to sum to 1 in all image pixels with
    nonzero probabilities. If a pixel has zeros for all label 
    probabilities, those zeros will be kept.
    '''
        
    if probabilities.shape[0]>1:
        probsum = probabilities.sum(axis=0, keepdims=True)
        probsum[probsum==0] = 1  # to avoid division by 0
        probabilities = probabilities/probsum
        
    elif probabilities.shape[0]==1:
        probabilities[probabilities>0] = 1
        
    return probabilities    

def imscale(image, scale=None, size=None, interpolation='linear'):
    '''
    Using opencv for resizing the image (swithching from skimage.transform).
    Give either ar scale (scaling factor) or desired image size. Treats layers
    along the first axis, so image of shape (5, 100, 120) resized with 
    scale=0.5 yields an image of shape (5, 50, 60).
    '''

    interpolations = {'linear':cv2.INTER_LINEAR, 'nearest':cv2.INTER_NEAREST}
    intpl = interpolations[interpolation]
    
    if scale is not None:
        size = tuple(int(sh*scale) for sh in image.shape[-2:])  # ignoring evt. layers
    size = (size[1], size[0])  # cv expects this?!
    if image.ndim==2:
        return cv2.resize(image, size, interpolation=intpl)
    elif image.ndim==3:
        return np.array([cv2.resize(im, size, interpolation=intpl) for im in image])
        
        
    