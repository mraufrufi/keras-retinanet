"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np

from .colors import label_color

import rasterio.plot
import matplotlib as mpl
from descartes import PolygonPatch
from matplotlib import pyplot as plt
import os

def draw_box(image, box, color, thickness=1):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=1):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=(255,165,0), label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    for a in annotations:
        label   = a[4]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)

def draw_ground_overlap(plot,ground_truth,ground_truth_tiles,projected_boxes,save_path):
    
    

    #set axis, 
    xmin,ymin,xmax,ymax=[int(x) for x in ground_truth[plot]["bounds"]]
    
    with rasterio.open(ground_truth_tiles[plot]) as src:
        
        rasterio.plot.show((src))
        ax = mpl.pyplot.gca()
        
        #Set axis with a bit of padding in meters
        ax.axis([xmin-5,xmax+5,ymin-5,ymax+5])               
        
        #Truth
        patches = [PolygonPatch(feature["geometry"]) for feature in ground_truth[plot]["data"]]
        collection=mpl.collections.PatchCollection(patches)
        collection.set_facecolor("none")    
        collection.set_edgecolor("blue")            
        ax.add_collection(collection)      
        
        #Predicted
        pred_patches = [PolygonPatch(feature) for feature in projected_boxes]
        collection=mpl.collections.PatchCollection(pred_patches)
        collection.set_facecolor("none")
        collection.set_edgecolor("red")            
        ax.add_collection(collection) 
        
        # zoom in to ground truth
        plt.savefig(os.path.join(save_path, '{}_overlay.png'.format(plot)),file_name=str(plot))    