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


def _read_annotations(csv_data_file,res):
    """ Read annotations from the csv_reader. Rescale origin by the resolution to get box coordinates with respect to cluster.
    """
    
    data=pd.read_csv(csv_data_file,index_col=0)    
    
    #Modify indices, which came from R, zero indexed in python
    data=data.set_index(data.index.values-1)
    
    #Modify Cluster indices, they start at 1 from R.
    data.Cluster=data.Cluster-1
    
    ##Create bounding coordinates with respect to the crop for each box
    #Rescaled to resolution of the cells.Also note that python and R have inverse coordinate Y axis, flipped rotation.
    data['origin_xmin']=(data['xmin']-data['cluster_xmin'])/res
    data['origin_xmax']=(data['xmin']-data['cluster_xmin']+ data['xmax']-data['xmin'])/res
    data['origin_ymin']=(data['cluster_ymax']-data['ymax'])/res
    data['origin_ymax']= (data['cluster_ymax']-data['ymax']+ data['ymax'] - data['ymin'])/res  
    
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

class OnTheFlyGenerator(generator.Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        config,
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
        
        #debug - plot images, based on config fiile
        self.plot_image=config['plot_image']
        
        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)
            
        #Read classes
        self.classes=_read_classes(csv_data_file)  
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key        

        self.rgb_tile_dir=config['rgb_tile_dir']
        self.rgb_res=config['rgb_res']
        
        #Read image data
        self.image_data=_read_annotations(csv_data_file,self.rgb_res)

        self.image_names = list(self.image_data.keys())

        super(OnTheFlyGenerator, self).__init__(**kwargs)
          
    def show(self,image,image_index):
        
        #Show bounding boxes on cropped image
        
        fig,ax = pyplot.subplots(1)
        ax.imshow(np.asarray(image))
        rows=self.image_data[image_index]
        for box in rows:            
            bottom_left=(box['origin_xmin'],box['origin_ymin'])
            height=box['origin_ymax']-box['origin_ymin']
            width=box['origin_xmax']-box['origin_xmin']
            rect = patches.Rectangle(bottom_left,width,height,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        pyplot.show()
        
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
        out_image=out_image/255
        
        #view image if needed on debug
        if self.plot_image:
            self.show(out_image, image_index)
        
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
            boxes[idx, 0] = float(annot['origin_xmin'])
            boxes[idx, 1] = float(annot['origin_ymin'])
            boxes[idx, 2] = float(annot['origin_xmax'])
            boxes[idx, 3] = float(annot['origin_ymax'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes

if __name__=="__main__":
    #construct a config
    config={}
    config['rgb_tile_dir']="/Users/ben/Documents/DeepForest/data/"
    config['rgb_res']=0.1
    config['plot_image']=True
    #path="/Users/ben/Documents/DeepForest/data/detection_OSBS_006.csv"
    path="/Users/ben/Documents/DeepForest/data/tmp/detection.csv"
    training_generator=OnTheFlyGenerator(csv_data_file=path,group_method="random",config=config)
    
    for x in np.arange(10):
        boxes=training_generator.next()
        print(boxes)
