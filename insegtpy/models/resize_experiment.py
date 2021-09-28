import skimage.transform
import skimage.io
import cv2 as cv
import time



image = skimage.io.imread('../../data/NT2_0001.png').astype(float)/255.0



#%%
s = 0.68

t = time.time()
for i in range(10):
    imsc = skimage.transform.rescale(image, s, preserve_range=True)
    imsz = skimage.transform.resize(imsc, image.shape, preserve_range=True)    
print(time.time()-t)



t = time.time()
for i in range(10):
    imsc_cv = cv.resize(image, [int(s*sh) for sh in image.shape], interpolation = cv.INTER_CUBIC)
    imsz_cv = cv.resize(imsc_cv, image.shape, interpolation = cv.INTER_LINEAR)    
print(time.time()-t)    

#%%

#print(abs(imsc - imsc_cv).max())

print(abs(imsc_cv_d - imsc_cv).max())


