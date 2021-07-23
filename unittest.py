# ---------------------------- Knowledge-based LiDAR Feature Extraction  ---------------------------- 
# external modules
import laspy
import glob
import os
import numpy as np
import pdal
from osgeo import gdal
import skimage.io

# customized modules
from lidar2feature import utils, knowledgebase

# the name of the processing data
las_name = 'C_37EZ1_7415_sample.las'

# the file path for the input data and prefix
data_dir = './data/knowledgebase/'
os.makedirs(data_dir, exist_ok=True)
input_las_path = data_dir + las_name
las_prefix = data_dir + os.path.splitext(las_name)[0]

# the file paths for intermedia files
class_tif_path = las_prefix + '_class.tif'
hag_tif_path = las_prefix + '_hag.tif'
intensity_tif_path = las_prefix + '_intensity.tif'

# the file path for the final result
result_tif_path = las_prefix + '_result.tif'

# convert LAS to provided classification raster
utils.las_to_tif(input_las_path, class_tif_path, dimension="classification", datatype='int16', resolution=1)

# read the classification raster as array
class_array = utils.read_tif(class_tif_path)

# convert LAS to height above ground (HAG) raster
utils.las_to_hag(input_las_path, hag_tif_path, resolution=1)

# read the height above ground (HAG) raster as array
hag_array = utils.read_tif(hag_tif_path)

# assure both arrays (class_array and hag_array) should have the same dimension
assert class_array.shape == hag_array.shape, "Both classification and height arrays above ground should have the same dimension"

# recode and execute knowledge-based classification
tree_height_threshold = 1
result_arary = knowledgebase.classification(class_array, hag_array, tree_height_threshold)

# export the classified result
utils.save_tif(class_tif_path, result_arary, result_tif_path)

# set the sliding window size. 
# noet: the large the size is the smoother the result will be. 
sliding_window_size = 3
result_arary = utils.majority_filter(result_arary, sliding_window_size)

# export the final filtered result
utils.save_tif(class_tif_path, result_arary, result_tif_path)
