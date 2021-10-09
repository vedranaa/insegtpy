import insegtpy.models.utils as utils
import numpy as np


class Segt():
    ''' Base class for image segmentation model.'''
        
    def process(self, labels, nr_classes=None):
        ''' Needs to be implemented for use with InSegtAnnotator.'''       
        pass
     
    def segment_new(self, image):
        ''' Needs to be implemented for processing a new image.'''
        pass
    
    

class Repeated(Segt):
    ''' Class providing repeated update.'''
       
    def __init__(self, segt, propagation_repetitions):
        self.onestep_segt = segt
        self.propagation_repetitions = propagation_repetitions
    
    def process(self, labels):
        nr_classes = labels.max()  # for many repetitions a label may dissapear
        probs = self.onestep_segt.process(labels, nr_classes)
               
        # If nr_repetitions>1, repeat
        for k in range(self.propagation_repetitions - 1):
            labels = utils.segment_probabilities(probs)
            probs = self.onestep_segt.process(labels, nr_classes)  
        
        return utils.normalize_to_one(probs)
    
    def segment_new(self, image):
        return self.onestep_segt.segment_new(image)
    
    
    
class Multiscale(Segt):
    ''' 
    General multiscale segmentation.
    
    Multiscale segmentation is a collection of segmentations (usually of the
    same type) which operate on up or down scaled images. The resulting
    probability is a (slightly modified) product of probabilities for each
    scale.
    '''
    def __init__(self, image, scales, model_init):
        
        self.scales = scales
        self.segt_list = [model_init(utils.imscale(image, s, 'linear')) for s in scales]
        self.aditive_parameter = 1e-10
    
    
    def process(self, labels, nr_classes=None):
        
        # for smaller scales a label may dissapear
        if nr_classes is None: nr_classes = labels.max()  
        
        if nr_classes==0:  # special treatment for empty labeling
            return np.zeros((0,) + labels.shape)
        else:
        
            probs = np.ones((1,1,1))  # shape will be changed by broadcasting      
            for segt, s in zip(self.segt_list, self.scales):
                labels_scaled = utils.imscale(labels, scale=s, 
                                              interpolation='nearest')
                probs_scaled = segt.process(labels_scaled, nr_classes)
                probs_this = utils.imscale(probs_scaled, 
                                size=labels.shape)
                probs = probs * (probs_this + self.aditive_parameter)
            return probs
            
    
    def segment_new(self, image):
        
        probs = np.ones((1,1,1))
        for segt, s in zip(self.segt_list, self.scales):
            image_scaled = utils.imscale(image, s)
            probs_scaled = segt.segment_new(image_scaled)
            probs = probs * (utils.imscale(probs_scaled, size=image.shape) + 
                                 self.aditive_parameter)
        return probs    

    

