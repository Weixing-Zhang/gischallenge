import numpy as np


def classification(class_array, hag_array, height=1):
    """
    Description: 
    The knowledgebase classification method use the build-in classification information
    of the LAS file. On top of that, the code does the following:
    (1) Find the areas without any LiDAR returns and classifies them as water because water
        absorb LiDAR and will not return any laser light
    (2) The code assumes that tree are taller that 1m but are not building and other 
        infrastructure

    -------------------------
    LAS Classification Codes:
    0: Never classified; 1: Unassigned; 2: Ground; 
    3: Low Vegetation; 4: Medium Vegetation; 5: High Vegetation; 
    6: Building; 
    7: Noise; 8: Model Key/Reserved; 
    9: Water; 10: Rail; 11: Road Surface; 
    12: Overlap/Reserverd; 
    13: Wire - Guard (Shield); 14: Wire - Conductor (Phase); 
    15: Transmission Tower; 16: Wire-Structure Connector; 17: Bridge Deck;
    18: High Noise; 
    19-255: Reserved; 

    Final Classification Codes:
    1: building; 2: tree; 3: water; 4: road; 5: infrastructure; 6: ground; 7: other
    -------------------------
    
    Inputs:
        class_array:
        hag_array:
    
    Outputs:
        result_arary:
        
    """
    # create base array
    result_arary = np.copy(class_array)

    ### RECODE DICTIONARY
    ### target classes:
    ### 1: building; 2: tree; 3: water; 4: road; 5: infrastructure; 6: ground; 7: other
    recode_dictionary = {"building-1":[6],
                        "tree-2":[4, 5],
                        "water-3":[9],
                        "road-4":[11],
                        "infrastructure-5":[10, 13, 14, 15, 16, 17],
                        "ground-6":[2, 3],
                        "other-7":[0, 1, 7, 8, 12, 18, 19]}

    ### RECODE 
    # refers to the class_array so the order of recoding does not matter
    for recoded_class in recode_dictionary:
        
        # recode based on its values
        recoded_class_value = int(recoded_class.split('-')[-1])   
        for original_class_value in recode_dictionary[recoded_class]:
            result_arary[class_array==original_class_value] = recoded_class_value
        
    # need to recode original 19 plus to other-7
    result_arary[class_array>19] = 7

    # change non return (-9,999) to water-3 because water does not have any return points
    result_arary[class_array==-9999] = 3

    ### CLASSIFY TREE BASED ON HEIGHT ABOUT GROUND
    # assume that tree are taller that 1m but are not building-1 and infrastructure-5
    result_arary[(hag_array>height) & ((result_arary!=1) & (class_array!=5))] = 2

    return result_arary
