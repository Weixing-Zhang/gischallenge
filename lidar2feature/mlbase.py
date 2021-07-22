from lidar2feature import external


def classification(ply_path, prefix):
    """ 
    Description: 
        Classify the points based on their calculated point-characteristics using a pre-trained 
        Bidirectional Long Short-Term Memory (BiLSTM) model. This function was kept here as a promise 
        to behave because I could not complete the complete all the procedure yet. 
    --------------------------------------
    Arguments:
        ply_path: the file path of the input .ply file
        prefix: the ply_path without file extension for the convenience of creating new file path
    Return:
        ply_path: the file path of the output .ply file
    """
    
    # call the predict function to classify all the point
    ply_path = external.predict_ply(ply_path, prefix)
    print(f"Completed classifying all points and stored them at {ply_path} ...")

    return ply_path
