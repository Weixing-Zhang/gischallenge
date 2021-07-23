import laspy
import pdal
import numpy as np
from osgeo import gdal
from scipy import stats
from scipy.ndimage import generic_filter

# lidar2feature.external module was developed by others
from lidar2feature.external import dict2ply


##################################################################
#  Process TIFF ('.tif') Imagery
##################################################################

def read_tif(tif_path):
    """
    Description:
        Read '.tif' image as an array
    --------------------------------------
    Arguments:
        tif_path: the file path of the '.tif' image
    Return:
        tif_array: an numpy array of the '.tif' image
    """

    refer = gdal.Open(tif_path)
    tif_array = refer.ReadAsArray()
    
    return tif_array


def save_tif(reference_path, input_array, tif_path):
    """
    Description: 
        Save an array as '.tif' image with the correct 
        cellsize and required spatial reference information
    --------------------------------------
    Arguments:
        reference_path: the file path of a '.tif' which the output will be expected to be
        input_array: the to-be-converted array
        tif_path: the file path to export '.tif' file
    Return: 
        None
    """

    # read basic information of image
    refer = gdal.Open(reference_path)
    cols = refer.RasterXSize
    rows = refer.RasterYSize
    band = refer.GetRasterBand(1)
    datatype = band.DataType
    
    # Output
    driver = gdal.GetDriverByName('GTiff')
    outDataset = driver.Create(tif_path, cols, rows, 1, datatype)

    # Set projection information
    geotransform = refer.GetGeoTransform()
    outDataset.SetGeoTransform(geotransform)
    proj = refer.GetProjection()
    outDataset.SetProjection(proj)
    
    # write array into image
    outDataset.GetRasterBand(1).WriteArray(input_array)

    print(f"Saved the image at {tif_path}")


def majority(input_array):
    """
    Description: 
        Get the mode value of an array.
    --------------------------------------
    Arguments:
        input_array: the sliding window array
    Return:
        mode.mode[0]: the mode value of the sliding window array
    """

    # acquire the mode value
    mode = stats.mode(input_array)

    return mode.mode[0]


def majority_filter(input_array, size):
    """
    Description: 
        Replaces cells in a raster based on the majority of their contiguous neighboring cells.
    --------------------------------------
    Arguments:
        input_array: the to-be-filtered array
        slide_size (default as 3): the size of the sliding windows of the filter
    Return:
        filtered_array: the filtered array
    """

    # execute the majority
    filtered_array = generic_filter(input_array, majority, size)

    print(f"Filterd the noise using a majority filter with a size of {size} majority filter")

    return filtered_array


##################################################################
#  Process LAS data
##################################################################

def las_to_tif(las_path, tif_path, dimension, datatype, resolution):
    """ 
    Description: 
        Covert point cloud in .las format using an interpolation algorithm 
        (i.e., Shepard’s inverse distance weighting) into a raster in .tif format 
    --------------------------------------
    Arguments:
        las_path: the file path of the input .las file
        tif_path: the file path of the output .tif raster file
        dimension: the dimension of the .las file to be converted to raster such as classification, intensity, etc.
        datatype: the datatype of cells of the .tif raster file
        resolution (meter): the cell size of the output .tif raster file
    Return:
        None
    """

    # make sure inputs are correct
    assert las_path.lower().endswith('.las'), "Pleasa make sure this is a .las file"

    # json script for using pdal pipeline to convert LAS to TIF 
    json = """
    [
        "%s",
        {
        "type":"writers.gdal",
        "filename":"%s",
        "dimension":"%s",
        "data_type":"%s",
        "output_type":"max",
        "resolution": %f
        }
    ]
    """%(las_path, tif_path, dimension, datatype, resolution)

    # execute the pipeline
    pipeline = pdal.Pipeline(json)
    pipeline.execute()
    
    print("Coverted the point cloud to a raster with built-in classification information ...")


def las_to_hag(las_path, tif_path, resolution):
    """
    Description: 
        Extract the height above ground of all non-ground points using delaunay filter.
        The filter creates a delaunay triangulation of the count ground points closest to the non-ground point.
    --------------------------------------
    Arguments:
        las_path: the file path of the input .las file
        tif_path: the file path of the output .tif raster file
    Return:
        none
    """

    # json script for using pdal pipeline to convert .las data to the height above ground (HAG) 
    json = """
    [
        "%s",
        {
            "type":"filters.hag_delaunay"
        },
        {
            "type":"filters.ferry",
            "dimensions":"HeightAboveGround=>Z"
        },
        {
            "type":"writers.gdal",
            "filename":"%s",
            "dimension":"Z",
            "data_type":"float",
            "output_type":"mean",
            "gdaldriver":"GTiff",
            "resolution": %f,
            "radius": 1
        }
    ]
    """%(las_path, tif_path, resolution)

    # execute the pipeline
    pipeline = pdal.Pipeline(json)
    pipeline.execute()

    print("Generated the heigh above ground raster...")


################################################################################
#  Process PIL (note: only the LAS related parts were developed by the author)
################################################################################

def las_to_ply(las_path, las_prefix, centering=True, scale=1):
    """ 
    Description: 
        Convert .las file to .ply file format so that the data can be consumed by the ML-based solution 
    --------------------------------------
    Arguments:
        las_path: the file path of the input .las file
        las_prefix: the las_path without file extension for the convenience of creating new file path
        centering (default True): decide if the x-,y-,and z- coordinates needs to be centered 
        scale (default 1): the target scale of the x-,y-,and z- coordinates; 1 means unchanged.
    Return:
        ply_path: the file path of the output .ply file
    """

    # read the input .las file
    las_file = laspy.read(las_path)  
    las_pnts = las_file.points
    
    ## In case someone wants to know what columns the .las file includes
    ## Please uncomment the following two lines to find them 
    # point_format = las_tile.point_format
    # print (list(point_format.dimension_names))

    # x-,y-,and z- coordinates need to be mutiplied by 0.001 to remain the correct decimals
    las_array = np.stack((las_pnts['X']*0.001,  
                          las_pnts['Y']*0.001,
                          las_pnts['Z']*0.001,
                          las_pnts['intensity'],
                          las_pnts['return_number'],
                          las_pnts['number_of_returns'],
                          las_pnts['classification']))

    # tranpose the array to match the PTS format
    las_array = np.transpose(las_array)

    # get the coordinates, features, and labels information
    coords = las_array[:, :3]
    features = las_array[:, 3:-1].astype(np.uint8)
    labels = las_array[:, -1].astype(np.uint8)

    # center the coordinates as default
    if centering:
        coords = coords - np.mean(coords, axis=0)

    # rescale the coordinates if the scale is not 1
    coords = (coords * scale).astype(np.float32)
    
    # construct the output .ply data
    ply_data = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "intensity": features[:, 0],
        "return_number": features[:, 1],
        "number_of_returns": features[:, 2],
        "labels": labels,
    }

    # save the centered point cloud in the .ply format
    ply_path = las_prefix + ".ply"

    if dict2ply(ply_data, ply_path):
        print(f"Completed converting the point cloud and saved them to {ply_path}")

    return ply_path
