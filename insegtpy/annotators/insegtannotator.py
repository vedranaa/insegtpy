#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Interactive segmentation annotator. 

This module contains the InSegtAnnotator class, which is a subclass of the 
Annotator class from the module annotator. InSegtAnnotator extends Annotator 
with the functionality for interactive segmentation. Segmentation is computed 
from annotations using a generic processing function.  


Use:
    Run from your environmend by passing InSegtAnnotator a grayscale uint8 
    image and a segmentation model with a processing function which given 
    labeling returns probabilities. 
    
Author: vand@dtu.dk, 2020
Created on Sun Oct 11 22:42:32 2020

GitHub:
   https://github.com/vedranaa/insegtpy
   
"""

import insegtpy.annotators.annotator as annotator
import numpy as np
import PyQt5.QtCore
import skimage

class InSegtAnnotator(annotator.Annotator):
    
    def __init__(self, image, model):
        '''
        Initializes InSegtAnnotator given an image and a segmentation model.

        Parameters
        ----------
        image : An image as a 2D array of dtype uint8.
        model : An object which has a processing function model.process.  
            Given an annotation (labeling), processing function returns a
            probability image. Annotation is given as a 2D array of  dtype 
            uint8, where 0 represents unlabeled pixels, and numbers
            1 to C represent labelings for diffeerent classes. (In current 
            impelmentation C<10.) A probability image is a 3D array with first
            two dimensions given by the shape of labeling, while the third
            dimension is given by the number of labels.
            
        '''
        
        imagePix = self.grayToPixmap(image)
        self.showProbabilities = 0  # __init__ needs this to set the title
        self.liveUpdate = True        
        super().__init__(imagePix.size()) # overwrittes self.imagePix
        self.imagePix = imagePix

        self.segmentationPix = PyQt5.QtGui.QPixmap(imagePix.width(), imagePix.height())
        self.segmentationPix.fill(self.color_picker(label=0, opacity=0))
        self.showImage = True
 
        self.probabilities = np.empty((0,self.imagePix.height(),self.imagePix.width()))
        self.probabilityPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), self.imagePix.height())
        self.probabilityPix.fill(self.color_picker(label=0, opacity=0))

        self.overlays = {0:'both', 1:'annotation', 2:'segmentation'}
        self.annotationOpacity = 0.3
        self.segmentationOpacity = 0.3
        self.model = model
        
        # check whether model has built-in callable attribute probToSeg 
        if not (hasattr(self.model, 'probToSeg') and callable(getattr(self.model, 'probToSeg'))):  
            self.model.probToSeg = self.probToSeg  # default
                
    
    # METHODS OVERRIDING ANNOTATOR METHODS:                   
    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.showImage:  # start by showing an image
            painter_display.drawPixmap(self.target, self.imagePix, self.source)
        if self.showProbabilities>0:  # show probilities
            painter_display.drawPixmap(self.target, self.probabilityPix, self.source)
        else:  # show annotations and/or segmentation
            if self.overlay != 1: # overlay 0 or 2
                    painter_display.drawPixmap(self.target, self.segmentationPix, self.source)
            if self.overlay != 2: # overlay 0 or 1
                    painter_display.drawPixmap(self.target, self.annotationPix, self.source)
        if self.showImage:  # show cursor if image is shown    
            painter_display.drawPixmap(self.target, self.cursorPix, self.source)
          
    
    def mouseReleaseEvent(self, event):
        """Segmentation is computed on mouse release."""
        if self.liveUpdate and (not self.activelyZooming):
            self.transformLabels()
            if self.showProbabilities>0:
                self.updateProbabilityPix()
        super().mouseReleaseEvent(event)
        self.update()
        
    
    def keyPressEvent(self, event):
        """Adding events to annotator"""   
        if event.key()==PyQt5.QtCore.Qt.Key_I:
            if self.showImage:  # not to react for consecutive keypress events while holding key          
                self.showImage = False
                self.update()
                self.showInfo('Turned off show image')              
        elif event.key()==PyQt5.QtCore.Qt.Key_P: 
            self.showProbabilities = (self.showProbabilities+1)%(self.probabilities.shape[0]+1)
            if self.showProbabilities:
                self.updateProbabilityPix()
                self.showInfo(f'Showing probability for label {self.showProbabilities}')
            else:
                self.showInfo('Not showing probabilities')
            self.setTitle()
        elif event.key()==PyQt5.QtCore.Qt.Key_L: 
            self.liveUpdate = not self.liveUpdate
            if self.liveUpdate:
                self.showInfo('Turned on live update')
                self.transformLabels()
                if self.showProbabilities:
                    self.updateProbabilityPix()
                self.update()
            else:
                self.showInfo('Turned off live update')
            self.setTitle()
        else:
            super().keyPressEvent(event)
            
            
  
    def keyReleaseEvent(self, event):
        """Adding events to annotator"""   
        if event.key()==PyQt5.QtCore.Qt.Key_I: # i
            self.showImage = True
            self.update()
            self.showInfo('Turned on show image')
        else:
            super().keyReleaseEvent(event)
    
    
    def setTitle(self):
        title = f'pen:{self.label}, width:{self.penWidth}, showing:'
        if self.showProbabilities:
            title += f'P{self.showProbabilities}'
        else:
            title += f'{self.overlays[self.overlay]}'
        title += ', live '
        if self.liveUpdate:
            title += 'on'
        else:
            title += 'off'
        self.setWindowTitle(title)
            
    
    # HELPING METHODS
    def transformLabels(self):
        """Transforming pixmap annotation to pixmap segmentation."""        
        annotations = self.pixmapToArray(self.annotationPix) # numpy RGBA: height x width x 4, values uint8      
        labels = self.rgbaToLabels(annotations) # numpy labels: height x width, values 0 to N uint8    
        self.probabilities = self.model.process(labels) # numpy labels: height x width, values 0 to N uint8
        
        segmentation = self.model.probToSeg(self.probabilities)
        segmentation_rgba = self.labelsToRgba(segmentation, 
                                              self.segmentationOpacity) # numpy RGBA: height x width x 4, values uint8  
        self.segmentationPix = self.rgbaToPixmap(segmentation_rgba)    # final pixmap    
        
        if(self.showProbabilities > self.probabilities.shape[0]): # removed a label for which we show probaiblity
            self.showProbabilities = 0
            self.showInfo('Not showing probabilities')
        
    
    def updateProbabilityPix(self):
        rgba = self.probabilityToRgba(self.probabilities[self.showProbabilities-1])
        self.probabilityPix = self.rgbaToPixmap(rgba)
            
    
    @staticmethod
    def probabilityToRgba(probabilityLayer):
        mask = (probabilityLayer > 0.5).astype(np.float)
        probabilityColor = np.asarray([2*(1-mask)*probabilityLayer + mask, 
                                       1-2*np.abs(probabilityLayer-0.5), 
                                       mask*(1-2*probabilityLayer) + 1,
                                       np.ones(mask.shape)*0.5]).transpose(1,2,0)
        return (255*probabilityColor).astype(np.uint8)
    
    
    @staticmethod
    def savePixmap(pixmap, filenamebase, gray):
        """Helping function for saving annotation and segmentation pixmaps."""
        pixmap.save(filenamebase + '_pixmap.png', 'png')
        rgba = InSegtAnnotator.pixmapToArray(pixmap) # numpy RGBA: height x width x 4, values uint8      
        skimage.io.imsave(filenamebase + '_rgb.png', rgba[:,:,:3], 
                          check_contrast=False)   
        labels = InSegtAnnotator.rgbaToLabels(rgba) # numpy labels: height x width, values 0 to N uint8    
        skimage.io.imsave(filenamebase + '_index.png', 30*labels, 
                          check_contrast=False) # 30*8 = 240<255     
        alpha = (rgba[:,:,3:].astype(np.float))/255
        overlay = gray[:,:,:3]*(1-alpha) + rgba[:,:,:3]*(alpha)
        skimage.io.imsave(filenamebase + '_overlay.png', 
                          overlay.astype(np.uint8), check_contrast=False)                 
         
    
    def saveOutcome(self):
        gray = self.pixmapToArray(self.imagePix) # numpy RGBA: height x width x 4, values uint8 
        skimage.io.imsave('gray.png', gray[:,:,:1], check_contrast=False)   
        self.savePixmap(self.annotationPix, 'annotations', gray)
        self.savePixmap(self.segmentationPix, 'segmentations', gray)
        self.showInfo('Saved annotations and segmentations in various data types')        
    
    
    helpText = (
        '<i>Help for InSegt Annotator</i> <br>' 
        '<b>KEYBOARD COMMANDS:</b> <br>' 
        '&nbsp; &nbsp; <b>1</b> to <b>9</b> changes pen label <br>' 
        '&nbsp; &nbsp; <b>0</b> eraser mode <br>' 
        '&nbsp; &nbsp; <b>&uarr;</b> and <b>&darr;</b> changes pen width <br>' 
        '&nbsp; &nbsp; <b>L</b> toggles live update <br>' 
        '&nbsp; &nbsp; <b>O</b> cycles between overlay settings <br>' 
        '&nbsp; &nbsp; <b>I</b> held down hides image <br>'
        '&nbsp; &nbsp; <b>P</b> cycles between probability views  <br>'
        '&nbsp; &nbsp; <b>Z</b> held down enables zoom <br>' 
        '&nbsp; &nbsp; <b>Z</b> pressed resets zoom <br>' 
        '&nbsp; &nbsp; <b>S</b> saves results <br>' 
        '&nbsp; &nbsp; <b>H</b> shows this help <br>' 
        '<b>MOUSE DRAG:</b> <br>' 
        '&nbsp; &nbsp; Draws annotation <br>' 
        '&nbsp; &nbsp; Zooms when zoom enabled')
    
    
    @classmethod
    def introText(cls, rich = True):
        if rich:
            return "<i>Starting InSegt Annotator</i> <br> For help, hit <b>H</b>"
        else:
            return "Starting InSegt Annotator. For help, hit 'H'."
        
     
    # for INSEGT, it is IMPORTANT that background is [0,0,0], otherwise rgbToLabels return wrong labels.
    # I therefore re-define collors, such that possible changes in annotator do not destroy InSegt
    # (and also I use numpy here)
    colors = np.array([
        [0, 0, 0], 
        [255, 0, 0], # label 1
        [0, 191, 0], # label 2
        [0, 0, 255], # etc
        [255, 127, 0],
        [0, 255, 191],
        [127, 0, 255],
        [191, 255, 0],
        [0, 127, 255],
        [255, 64, 191]], dtype=np.uint8)          

    # METHODS TRANSFORMING BETWEEN NUMPY (RGBA AND LABELS) AND QT5 DATA TYPES:
    @classmethod
    def rgbaToLabels(cls,rgba):
        """RGBA image to labels from 0 to N. Uses colors. All numpy."""    
        rgb = rgba.reshape(-1,4)[:,:3] # unfolding and removing alpha channel
        dist = np.sum(abs(rgb.reshape(-1,1,3).astype(np.int16) 
                - cls.colors.reshape(1,-1,3).astype(np.int16)), axis=2) # distances to pre-defined colors
        labels = np.empty((rgb.shape[0],), dtype=np.uint8)
        np.argmin(dist, axis=1, out=labels) # label given by the smallest distances
        labels = labels.reshape(rgba.shape[:2]) # folding back
        return(labels)
    
    @classmethod
    def labelsToRgba(cls, labels, opacity=1):
        """Labels from 0 to N to RGBA. Uses colors. All numpy."""
        rgb = cls.colors[labels,:]
        a = (255*opacity*(labels>0)).astype(np.uint8) # alpha channel
        a.shape = a.shape + (1,)
        rgba = np.concatenate((rgb, a), axis=2)
        return(rgba)
    
    @staticmethod
    def pixmapToArray(qpixmap):
        """Qt pixmap to np array. Assumes an 8-bit RGBA pixmap."""
        qimage = qpixmap.toImage().convertToFormat(PyQt5.QtGui.QImage.Format_RGBA8888)
        buffer = qimage.constBits()
        buffer.setsize(qpixmap.height() * qpixmap.width() * 4) # 4 layers for RGBA
        rgba = np.frombuffer(buffer, np.uint8).reshape((qpixmap.height(), 
                qpixmap.width(), 4))
        return rgba.copy()
    
    @staticmethod
    def rgbaToPixmap(rgba):
        """Np array to Qt pixmap. Assumes an 8-bit RGBA image."""
        rgba = rgba.copy()
        qimage = PyQt5.QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], 
                                    PyQt5.QtGui.QImage.Format_RGBA8888)
        qpixmap = PyQt5.QtGui.QPixmap(qimage)
        return qpixmap
    
    @staticmethod
    def grayToPixmap(gray):
        """Uint8 grayscale image to Qt pixmap via RGBA image."""
        rgba = np.tile(gray,(4,1,1)).transpose(1,2,0)
        rgba[:,:,3] = 255
        qpixmap = InSegtAnnotator.rgbaToPixmap(rgba)
        return qpixmap
    
    @staticmethod
    def probToSeg(probabilities):
        '''Probabilities to segmentation using max-prob approach.
        '''
        segmentation = np.zeros(probabilities.shape[1:], dtype=np.uint8)  # max 255 labels
        if probabilities.shape[0]>1:
            p = np.sum(probabilities, axis=0)
            np.argmax(probabilities, axis=0, out=segmentation)
            segmentation += 1
            segmentation[p==0] = 0
        elif probabilities.shape[0]==1:
            segmentation[probabilities[0]>0] = 1
        return segmentation
        

def insegt(image, processing_function):
    '''
    image : grayscale image given as (r,c) numpy array of type uint8 
    processing_function : a functiobn which given label image of size (r,c)
        returns segmentation image
    '''
    app = PyQt5.QtWidgets.QApplication([])
    ex = InSegtAnnotator(image, processing_function)
    ex.show()
    app.exec()
    return(ex)
    
   

    
    
    
    