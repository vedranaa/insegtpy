import numpy as np
import cv2

def get_gauss_feat_im(image, sigma, dtype='float32', output=None):
    """Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: standard deviation for Gaussian derivatives.
        dtype: data type for image and features. Default is float32.
        output: optional output array. If provided, it must have shape
            (15,r,c) and dtype matching image.
    Returns:
        imfeat: a 3D array of size (15,r,c) with a 15-dimentional feature
            vector for every image pixel.
    Author: vand@dtu.dk, 2020; niejep@dtu.dk, 2023
    """

    # Ensure image is float32.
    # This data type is often much faster than float64.
    if image.dtype != dtype:
        image = image.astype(dtype)

    # Create kernel array.
    s = np.ceil(sigma * 4)
    x = np.arange(-s, s + 1).reshape((-1, 1))

    # Create Gaussian kernels.
    g = np.exp(-x**2 / (2 * sigma**2))
    g /= np.sum(g)
    g = g.astype(image.dtype)  # Make same type as image.
    dg = -x / (sigma**2) * g
    ddg = -1 / (sigma**2) * g - x / (sigma**2) * dg
    dddg = -2 / (sigma**2) * dg - x / (sigma**2) * ddg
    ddddg = -2 / (sigma**2) * ddg - 1 / (sigma**2) * ddg - x / (sigma**2) * dddg

    # Create image feature arrays and temporary array. Features are stored
    # on the first access for fast direct write of values in filter2D.
    if output is None:
        imfeat = np.zeros((15, ) + image.shape, dtype=image.dtype)
    else:
        imfeat = output
    imfeat_tmp = np.zeros_like(image)

    # Extract features. Order is a bit odd, as original order has been
    # kept even though calculation order has been updated. We use the tmp
    # array to store results and avoid redundant calcalations. This
    # reduces calls to filter2D from 30 to 20. Results are written
    # directly to the destination array.
    cv2.filter2D(image, -1, g, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[0])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[2])
    cv2.filter2D(imfeat_tmp, -1, ddg.T, dst=imfeat[5])
    cv2.filter2D(imfeat_tmp, -1, dddg.T, dst=imfeat[9])
    cv2.filter2D(imfeat_tmp, -1, ddddg.T, dst=imfeat[14])

    cv2.filter2D(image, -1, dg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[1])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[4])
    cv2.filter2D(imfeat_tmp, -1, ddg.T, dst=imfeat[8])
    cv2.filter2D(imfeat_tmp, -1, dddg.T, dst=imfeat[13])

    cv2.filter2D(image, -1, ddg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[3])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[7])
    cv2.filter2D(imfeat_tmp, -1, ddg.T, dst=imfeat[12])

    cv2.filter2D(image, -1, dddg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[6])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[11])

    cv2.filter2D(image, -1, ddddg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[10])

    return imfeat

    
class GaussFeatureExtractor:
    # A callable class for extracting Gaussian features while storing normalization
    def __init__(self, sigmas=[1,2,4]):
        '''
        Extractor which maintains parameters for extracting Gaussian features.
        After the first feature extraction with True flag 
        update_normalization_parameters, it will normalize all subsequently
        extracted features.
        
        Parameters:
            sigmas: a list with standard deviation for Gaussian features.
        
        Author: vand@dtu.dk, abda@dtu.dk, 2021
        '''
        if type(sigmas) is not list: sigmas = [sigmas]
        self.sigmas = sigmas
        self.normalization_count = int(0)
        self.normalization_means = None
        self.normalization_stds = None

    def __call__(self, image, update_normalization=False, normalize=True, dtype='float32'):
        '''Extract Gaussian derivative feaures for every image pixel.
        
        Arguments:
            image: a 2D image, shape (r,c).
            update_normalization: a boolean flag indicating whether 
                normalization parameters should be updated.
            normalize: a boolean flag indicating whether to normalize.
                Requires that normalization_parameters exist.
            dtype: data type for the features. Default is float32.
        Returns:
            features, 3D array of shape (n,r,c)
         
        Author: vand@dtu.dk, abda@dtu.dk, 2021
        '''           
                
        # Compute features.
        features = np.zeros((len(self.sigmas), 15) + image.shape, dtype=dtype)
        for i, sigma in enumerate(self.sigmas):
            get_gauss_feat_im(image, sigma, output=features[i])
        features = features.reshape((-1,)+image.shape)
        
        # If needed, update normalization_parameters.                
        if update_normalization:
            means = np.mean(features, axis=(1,2))
            stds = np.std(features, axis=(1,2))           
            
            if self.normalization_count==0:
                self.normalization_means = means
                self.normalization_stds = stds   
            else:
                w = self.normalization_count/(self.normalization_count+1)
                self.normalization_means *= w
                self.normalization_means += (self.normalization_count+1) * means
                self.normalization_stds **= 2   
                self.normalization_stds *= w
                self.normalization_stds += (self.normalization_count+1) * (stds**2)
                self.normalization_stds **= 0.5
            self.normalization_count += 1 
        
        # If needed, normalize
        if normalize and self.normalization_count>0:
            features -= self.normalization_means.reshape((-1,1,1))
            features *= (1/self.normalization_stds).reshape((-1,1,1))
            
        return features
        
        
        
if __name__ == '__main__':
   
    # Example use 
    import skimage.io
    import matplotlib.pyplot as plt
    
    image_0 = skimage.io.imread('../../data/carbon0103.png').astype(float)/255
    image_1 = skimage.io.imread('../../data/carbon0969.png').astype(float)/255
    
    extractor = GaussFeatureExtractor(sigmas = [1,2,5,10])
    
    # First use
    features_0 = extractor(image_0, update_normalization=True)
    # Subsequent uses may update or not
    features_1 = extractor(image_1, update_normalization=False)
    
    l = range(0, 60, 10)
    
    fig, ax = plt.subplots(2, len(l))
    for i in range(len(l)):
        ax[0, i].imshow(features_0[l[i]])
        ax[1, i].imshow(features_1[l[i]])
        
    
        
    
        
        
        
        