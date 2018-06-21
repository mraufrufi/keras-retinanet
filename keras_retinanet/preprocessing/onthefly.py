"""
On the fly generator. Crop out portions of a large image, and pass boxes and annotations. This follows the csv_generator template. Satifies the format in generator.py
"""
import pandas as pd

from keras_retinanet.preprocessing import generator
from keras_retinanet.utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path

#Cropping on the fly
from rasterio.mask import mask
import rasterio
from matplotlib import pyplot 
import matplotlib.patches as patches

import warnings
warnings.filterwarnings("error")

def _read_classes(csv_data_file):
    """ Parse the classes file given by csv_reader.
    """
    
    # Read in data csv
    data=pd.read_csv(csv_data_file,index_col=0)
    
    #Modify indices, which came from R, zero indexed in python
    data=data.set_index(data.index.values-1)
    
    #Modify Cluster indices, they start at 1 from R.
    data.Cluster=data.Cluster-1    
    
    #Get unique classes
    uclasses=data.loc[:,['label','numeric_label']].drop_duplicates()
    
    # Define classes 
    classes = {}
    for index, row in uclasses.iterrows():
        classes[row.label] = row.numeric_label
    
    return(classes)


def _read_annotations(csv_data_file):
    """ Read annotations from the csv_reader.
    """
    
    data=pd.read_csv(csv_data_file,index_col=0)    
    
    #Modify indices, which came from R, zero indexed in python
    data=data.set_index(data.index.values-1)
    
    #Modify Cluster indices, they start at 1 from R.
    data.Cluster=data.Cluster-1
    
    ##Create bounding coordinates with respect to the crop for each box
    for box in data.iterrows():
        box['origin_xmin']=box['xmin']-box['cluster_xmin']
        box['origin_xmax']=box['cluster_xmax']-box['xmax']
        box['origin_ymin']=box['ymax']-box['cluster_ymax']
        box['origin_ymax']=(box['cluster_ymax']-box['cluster_ymin'])-box['ymin']    
    
    result={}
    
    for index,row in data.iterrows():
        
        #check if new cluster
        if row["Cluster"] not in result:
            result[row["Cluster"]] = []
        
        #append annotations
        result[row["Cluster"]].append(row.to_dict())
        
    return(result)

#cropping rgb data
def data2geojson(row):
    '''
    Convert a pandas row into a polygon bounding box
    ''' 
    
    features={"type": "Polygon",
         "coordinates": 
         [[(float(row["cluster_xmin"]),float(row["cluster_ymin"])),
             (float(row["cluster_xmax"]),float(row["cluster_ymin"])),
             (float(row["cluster_xmax"]),float(row["cluster_ymax"])),
             (float(row["cluster_xmin"]),float(row["cluster_ymax"])),
             (float(row["cluster_xmin"]),float(row["cluster_ymin"]))]]}       
    
    return features

class CSVGenerator(generator.Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        base_dir=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)
            
        #Read classes
        self.classes=_read_classes(csv_data_file)
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key        

        ####TO DO !!! Change hardcoded tile to read .config.yml
        self.rgb_tile_dir="/Users/ben/Documents/TreeSegmentation/data/2017/RGB/"
        
        #Read image data
        self.image_data=_read_annotations(csv_data_file)

        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)
          
    def show(self,image,image_index,boxes):
        
        fig,ax = pyplot.subplots(1)
        ax.imshow(np.asarray(image))
        rows=self.image_data[image_index]
        for box in rows:            
            bottom_left=(box['origin_xmin'],box['origin_ymin'])
            height=box['origin_ymax']-box['origin_ymin']
            width=box['origin_xmax']-box['origin_xmin']
            rect = patches.Rectangle(bottom_left,height,width,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        
        
    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]
    
    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        
        """
        row=self.image_data[image_index]
        
        #Crop cluster, all cluster coordinates the same within a image group
        
        #generate polygon
        #create polygon from bounding box from the first row.
        features=data2geojson(row[0])
        
        #crop and return image, all rgb_paths are the same
        with rasterio.open(self.rgb_tile_dir + row[0]['rgb_path']) as src:
            out_image, out_transform = mask(src, [features], crop=True)
            
        #color channel should be last, tensorflow convention?
        out_image=np.moveaxis(out_image, 0, -1)     
        
        #scale to 0-255
        out_image=out_image/1000
        
        #TODO is the BGR or RGB? see read_image_bgr in util
        return out_image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        
        path   = self.image_names[image_index]
        annots = self.image_data[path]
        boxes  = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = annot['label']
            boxes[idx, 0] = float(annot['xmin'])
            boxes[idx, 1] = float(annot['ymin'])
            boxes[idx, 2] = float(annot['xmax'])
            boxes[idx, 3] = float(annot['ymax'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes

if __name__=="__main__":
    
    path="/Users/ben/Documents/DeepForest/data/detection_OSBS_006.csv"
    training_generator=CSVGenerator(csv_data_file=path,group_method="random")
    boxes=training_generator.next()
    print(boxes)
