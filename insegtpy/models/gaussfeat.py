import numpy as np
import cv2


def get_gauss_feat_im(image, sigma):
    """Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: standard deviation for Gaussian derivatives.
    Returns:
        imfeat: a 3D array of size (15,r,c) with a 15-dimentional feature
            vector for every image pixel.
    Author: vand@dtu.dk, 2020
    """
      
    s = np.ceil(sigma*4)
    x = np.arange(-s,s+1).reshape((-1,1));

    g = np.exp(-x**2/(2*sigma**2));
    g /= np.sum(g);
    dg = -x/(sigma**2)*g;
    ddg = -1/(sigma**2)*g - x/(sigma**2)*dg;
    dddg = -2/(sigma**2)*dg - x/(sigma**2)*ddg;
    ddddg = -2/(sigma**2)*ddg - 1/(sigma**2)*ddg - x/(sigma**2)*dddg;
    
    imfeat = np.empty((15,) + image.shape)
    imfeat[0] = cv2.filter2D(cv2.filter2D(image,-1,g),-1,g.T)
    imfeat[1] = cv2.filter2D(cv2.filter2D(image,-1,dg),-1,g.T)
    imfeat[2] = cv2.filter2D(cv2.filter2D(image,-1,g),-1,dg.T)
    imfeat[3] = cv2.filter2D(cv2.filter2D(image,-1,ddg),-1,g.T)
    imfeat[4] = cv2.filter2D(cv2.filter2D(image,-1,dg),-1,dg.T)
    imfeat[5] = cv2.filter2D(cv2.filter2D(image,-1,g),-1,ddg.T)
    imfeat[6] = cv2.filter2D(cv2.filter2D(image,-1,dddg),-1,g.T)
    imfeat[7] = cv2.filter2D(cv2.filter2D(image,-1,ddg),-1,dg.T)
    imfeat[8] = cv2.filter2D(cv2.filter2D(image,-1,dg),-1,ddg.T)
    imfeat[9] = cv2.filter2D(cv2.filter2D(image,-1,g),-1,dddg.T)
    imfeat[10] = cv2.filter2D(cv2.filter2D(image,-1,ddddg),-1,g.T)
    imfeat[11] = cv2.filter2D(cv2.filter2D(image,-1,dddg),-1,dg.T)
    imfeat[12] = cv2.filter2D(cv2.filter2D(image,-1,ddg),-1,ddg.T)
    imfeat[13] = cv2.filter2D(cv2.filter2D(image,-1,dg),-1,dddg.T)
    imfeat[14] = cv2.filter2D(cv2.filter2D(image,-1,g),-1,ddddg.T)
    
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

    def __call__(self, image, update_normalization=False, normalize=True):
        '''Extract Gaussian derivative feaures for every image pixel.
        
        Arguments:
            image: a 2D image, shape (r,c).
            update_normalization: a boolean flag indicating whether 
                normalization parameters should be updated.
            normalize: a boolean flag indicating whether to normalize.
                Requires that normalization_parameters exist.
        Returns:
            features, 3D array of shape (n,r,c)
         
        Author: vand@dtu.dk, abda@dtu.dk, 2021
        '''           
                
        # Compute features.
        features = []
        for sigma in self.sigmas:
            features.append(get_gauss_feat_im(image, sigma))
        features = np.asarray(features).reshape((-1,)+image.shape)
        
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
        
    
        
    
        
        
        
        