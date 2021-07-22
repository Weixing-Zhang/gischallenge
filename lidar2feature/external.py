"""
IMPORTANT NOTES from the repository author: Weixing Zhang
All the code in external.py was acquired from https://github.com/theobdt/aerial_pc_classification and organized into the following format.
"""

#
# Original Author: Hugues THOMAS
# Date: 10/02/2017
# Required Modules
import os
import sys
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import spatial
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as skm


# trained model
MODEL = './configs/trained_model'

# config for the characteristics extraction
CONFIG = "./configs/config_characteristics_extraction.yaml"


##################################################################
#  Predict the class of point cloud
##################################################################

def predict_ply(path_ply, ply_prefix):
        
    path_ckpt = MODEL

    NAMES_9 = [
        "Powerline",
        "Low veg.",
        "Imp. surf.",
        "Car",
        "Fence",
        "Roof",
        "Facade",
        "Shrub",
        "Tree",
    ]

    NAMES_4 = ["GLO", "Roof", "Facade", "Vegetation"]

    # check usable devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Recommend to install the CUDA toolkit to run the model on GPU if you intend to process a large dataset ... ")
    print(f"* device in use : {device}")

    # Load checkpoint (i.e., the trained model)
    print(f"* trained model {path_ckpt} ...")
    path_config = os.path.join(path_ckpt, "config.yaml")
    path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
    checkpoint = torch.load(path_ckpt_dict, map_location=device)

    # Load model config
    with open(path_config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    n_features = len(config["data"]["features"])
    n_classes = 4
    if config["data"]["all_labels"]:
        n_classes = 9
    print(f"* number of classes: {n_classes}\n")

    print("Loading model ...")
    model = BiLSTM(n_features, n_classes, **config["network"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # load the input ply file
    print(f"\nProcessing file: {path_ply} ...")
    print("\nPreparing dataloader ...")
    dataset = AerialPointDataset(path_ply, **config["data"])
    loader = DataLoader(
        dataset=dataset,
        batch_size=1000,
        num_workers=4,
        shuffle=False,
    )

    # Create and fill point cloud field
    data = ply2dict(path_ply)
    true_labels = data["labels"]
    names = NAMES_9

    # in the 4-labels case
    if not config["data"]["all_labels"]:
        true_labels = convert_labels(true_labels).astype(np.int32)
        names = NAMES_4

    n = len(true_labels)
    predictions = -np.ones(n, dtype=np.int32)
    raw_predictions = predict(loader, len(dataset), model).astype(np.int32)
    predictions[dataset.index] = raw_predictions
    errors = predictions != true_labels
    data["predictions"] = predictions
    data["errors"] = errors.astype(np.uint8)
    data["labels"] = true_labels

    # Save point cloud
    path_prediction = ply_prefix + '_result.ply'
    if dict2ply(data, path_prediction):
        pass

    return path_prediction


def predict(loader, len_dataset, model):
    print ("")
    print ("* total number of points:", len_dataset)
    predictions = torch.empty(len_dataset, dtype=torch.int32, device=device)
    
    with torch.no_grad():
        start = 0
        for (sequence, label) in tqdm(loader, desc="* Processing point cloud"):
            sequence = sequence.to(device)
            label = label.to(device)

            # compute predicted classes
            output = model(sequence)
            classes = torch.max(output, 1).indices

            # fill predictions
            seq_len = sequence.shape[0]
            predictions[start : start + seq_len] = classes
            start += seq_len

    return predictions.cpu().numpy()


def evaluate(y_true, y_pred, names):
    labels = np.arange(len(names))

    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    totals = np.sum(cm, axis=1)
    cm = np.hstack((cm, totals.reshape(-1, 1)))
    totals_cols = np.sum(cm, axis=0, keepdims=True)
    cm = np.vstack((cm, totals_cols, totals_cols))

    metrics = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=labels
    )
    metrics = 100 * np.vstack(metrics[:-1]).T
    avg_metrics = np.mean(metrics, axis=0)
    weighted_avg_metrics = totals @ metrics / np.sum(totals)
    metrics = np.vstack((metrics, avg_metrics, weighted_avg_metrics))

    all_data = np.hstack((cm, metrics))

    cols_int = names + ["Total"]
    cols_float = ["Precision", "Recall", "F1-score"]

    idx = names + ["Total/Avg", "Total/Weighted Avg"]
    df = pd.DataFrame(data=all_data, columns=cols_int + cols_float, index=idx)
    df[cols_int] = df[cols_int].astype(int)
    return df


def write_metrics(path_prediction, filename, df):
    filename = filename.split(".")[0]
    path_metrics = os.path.join(path_prediction, "metrics")
    os.makedirs(path_metrics, exist_ok=True)

    path_tex = os.path.join(path_metrics, f"{filename}.tex")
    path_txt = os.path.join(path_metrics, f"{filename}.txt")
    print(path_tex)

    # write tex file
    column_format = "|l|" + (df.shape[1] - 4) * "r|" + "|r||r|r|r|"
    with open(path_tex, "w") as f:
        f.write(
            df.to_latex(
                float_format="{:0.2f}".format, column_format=column_format,
            )
        )

    with open(path_txt, "w") as f:
        df.to_string(f)
    print(f"* Metrics written to: {path_tex} and {path_tex}")


##################################################################
#  BiLSTM Model
##################################################################

SIZE_RELATION_TYPE = {0: 0, 1: 1, 2: 3, 3: 4, 4: 3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(
        self, n_features, num_classes, hidden_size, num_layers, relation_type=1
    ):
        super(BiLSTM, self).__init__()
        self.relation_type = relation_type

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        try:
            size_relation_vector = SIZE_RELATION_TYPE[relation_type]
        except KeyError:
            print(f"Relation type '{self.relation_type}' not recognized")
            return

        input_size = n_features + size_relation_vector
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection


    def forward(self, x, debug=False):

        if debug:
            print("\ninput size")
            print(x.shape)
        x = self.relation_vectors(x)

        if debug:
            print("after transform")
            print(x.shape)

        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        if debug:
            print("batch size")
            print(batch_size)
            print("hidden")
            print(hidden[0].shape)

        # Propagate input through LSTM :
        out, hidden = self.lstm(x, hidden)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        if debug:
            print("output size")
            print(out.shape)

        return out


    def init_hidden(self, batch_size):
        # initialization of hidden states
        # shape (num_layers * num_directions, batch, hidden_size)
        hidden = (
            Variable(
                torch.zeros(
                    self.num_layers * 2,
                    batch_size,
                    self.hidden_size,
                    device=device,
                )
            ),
            Variable(
                torch.zeros(
                    self.num_layers * 2,
                    batch_size,
                    self.hidden_size,
                    device=device,
                )
            ),
        )
        return hidden


    def relation_vectors(self, x):
        coords = x[:, :, :3]
        # shape : (batch_size, seq_len, 3)

        # no relation
        if self.relation_type == 0:
            return x[:, :, 3:]

        # distances only
        if self.relation_type == 1:
            diff = coords - coords[:, 0:1, :]
            distances = torch.sum(diff ** 2, dim=2, keepdim=True)
            return torch.cat((distances, x[:, :, 3:]), dim=2)

        # centered coords
        elif self.relation_type == 2:
            diff = coords - coords[:, 0:1, :]
            return torch.cat((diff, x[:, :, 3:]), dim=2)

        # centered coords + distances
        elif self.relation_type == 3:
            diff = coords - coords[:, 0:1, :]
            distances = torch.sum(diff ** 2, dim=2, keepdim=True)
            return torch.cat((diff, distances, x[:, :, 3:]), dim=2)

        # decentralized coords
        elif self.relation_type == 4:
            decentralized = coords - torch.min(coords, dim=0, keepdim=True)[0]
            return torch.cat((decentralized, x[:, :, 3:]), dim=2)


##################################################################
#  Data Loader
##################################################################

class AerialPointDataset(Dataset):
    def __init__(self, input_file, features, n_neighbors, all_labels=False):
        "Initialization"
        data = ply2dict(input_file)
        try:
            all_features = ["x", "y", "z"] + features
            X = np.vstack([data[f] for f in all_features]).T
        except KeyError:
            print(f"ERROR: Input features {features} not recognized")
            return
        labels = data["labels"]

        self.index = np.arange(X.shape[0])
        if not all_labels:
            X, labels = self.filter_labels(X, labels)

        self.X = torch.from_numpy(X)
        self.labels = torch.from_numpy(labels)
        self.n_samples = self.labels.shape[0]
        tree = KDTree(self.X[:, :3])
        _, self.neighbors_idx = tree.query(
            self.X[:, :3], k=n_neighbors, sort_results=True
        )


    def filter_labels(self, X, labels):
        new_labels = convert_labels(labels)
        mask = new_labels >= 0
        self.index = self.index[mask]
        return X[mask], new_labels[mask]


    def __getitem__(self, index):
        point = self.X[index].view(1, -1)
        neighbors = self.X[self.neighbors_idx[index]]
        sequence = torch.cat((point, neighbors), 0)

        return sequence, self.labels[index]


    def __len__(self):
        return self.n_samples


def convert_labels(labels):
    """Convert 9-labels to 4-labels as follows:
    0 Powerline              -> -1 Other
    1 Low vegetation         -> 0 GLO
    2 Impervious surfaces    -> 0 GLO
    3 Car                    -> -1 Other
    4 Fence/Hedge            -> -1 Other
    5 Roof                   -> 1 Roof
    6 Facade                 -> 2 Facade
    7 Shrub                  -> 3 Vegetation
    8 Tree                   -> 3 Vegetation
    """
    LABELS_MAP = {0: -1, 1: 0, 2: 0, 3: -1, 4: -1, 5: 1, 6: 2, 7: 3, 8: 3}
    return np.vectorize(LABELS_MAP.get)(labels)


##################################################################
#  Computer Characteristics
##################################################################

AVAILABLE_STEPS = [
"descriptors",
"region_growing",
"ground_extraction",
"ground_rasterization",
"height_above_ground",
]

def compute_characteristics(path_ply, ply_prefix):
    
    # Load config
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    print("This process can take time when processing a large .las file ... ")

    steps_params = {step: config[step] for step in AVAILABLE_STEPS}

    data = ply2dict(path_ply)
    coords = np.vstack((data["x"], data["y"], data["z"])).T

    grid_ground_3d = None

    for (step, params) in steps_params.items():
        if step == "descriptors":
            print("\nComputing local descriptors ...")
            all_descriptors = compute_descriptors(coords, **params)
            data.update(all_descriptors)

        if step == "region_growing":
            print("\nComputing regions ...")
            normals = np.vstack((data["nx"], data["ny"], data["nz"])).T
            params_copy = params.copy()

            descriptor_selected = params_copy.pop("descriptor")
            print(
                "* descriptor selected : "
                f"{'min' if params['minimize'] else 'max'} "
                f"{descriptor_selected}"
            )
            print(f"* thresholds : {params['thresholds']}")
            print(f"* radius : {params['radius']}")
            try:
                descriptor_vals = data[descriptor_selected]
                region_labels = multi_region_growing(coords, normals, descriptor_vals, **params_copy)

                data["regions"] = region_labels
            except KeyError:
                print(
                    f"Descriptor '{descriptor_selected}' has not been computed"
                    ", run 'python3 compute_features.py --descriptors "
                    f"{descriptor_selected}'"
                )
                sys.exit(-1)

        if step == "ground_extraction":
            print("\nExtracting ground from regions ...")
            region_labels = data["regions"]
            ground_mask = stitch_regions(
                coords, region_labels, **params
            )

            ground_only = {
                field: data[field][ground_mask] for field in list(data.keys())
            }

            data["ground"] = ground_mask.astype(np.uint8)
            path_ground = ply_prefix + '_ground_only.ply'
            if dict2ply(ground_only, path_ground):
                print(f"* PLY ground file successfully saved to {path_ground}")

        if step == "ground_rasterization":
            print("\nComputing ground rasterization ...")

            ground_mask = data["ground"].astype(bool)
            grid_ground_3d = rasterize_ground(
                coords, ground_mask, **params
            )

            ground_rasterized = {
                "x": grid_ground_3d[:, 0],
                "y": grid_ground_3d[:, 1],
                "z": grid_ground_3d[:, 2],
                "ground_altitude": grid_ground_3d[:, 2],
            }

            path_rasterized = ply_prefix + '_ground_rasterized.ply'
            if dict2ply(ground_rasterized, path_rasterized):
                print(
                    "* Completed the ground rasterized file and saved to "
                    f"{path_rasterized}"
                )

        if step == "height_above_ground":
            print("\nComputing height above ground ...")
            if grid_ground_3d is None:
                path_rasterized = ply_prefix + '_ground_rasterized.ply'
                print(f"* Loading rasterized ground : {path_rasterized}")
                ground_rasterized = ply2dict(path_rasterized)
                grid_ground_3d = np.vstack(
                    (
                        ground_rasterized["x"],
                        ground_rasterized["y"],
                        ground_rasterized["z"],
                    )
                ).T

            ground_mask = data["ground"].astype(bool)
            heights = height_above_ground(coords, ground_mask, grid_ground_3d)
            data["height_above_ground"] = heights

    # saving data
    path_output = ply_prefix + '_features.ply'
    if dict2ply(data, path_output):
        print(f"\nCompleted comptuing the characteristics and saved to {path_output}")

    return path_output


##################################################################
#  PLY files reader/writer
##################################################################

# Define PLY types
ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "b1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)


# Numpy reader format
valid_formats = {
    "ascii": "",
    "binary_big_endian": ">",
    "binary_little_endian": "<",
}


# Functions
def parse_header(plyfile, ext):

    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()
        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def read_ply(filename):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, "rb") as plyfile:

        # Check if the file start with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start whith the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # Parse header
        num_points, properties = parse_header(plyfile, ext)

        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append("element vertex %d" % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append("property %s %s" % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension
        will be appended to the file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of
        numpy arrays or a tuple of numpy arrays. Each 1D numpy array and each
        column of 2D numpy arrays are considered as one field.

    field_names : list
        the name of each fields as a list of strings. Has to be the same length
        as the number of fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = (
        list(field_list)
        if (type(field_list) == list or type(field_list) == tuple)
        else list((field_list,))
    )
    for i, field in enumerate(field_list):
        if field is None:
            print("WRITE_PLY ERROR: a field is None")
            return False
        elif field.ndim > 2:
            print("WRITE_PLY ERROR: a field have more than 2 dimensions")
            return False
        elif field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print("wrong field dimensions")
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print("wrong number of field names")
        return False

    # Add extension if not there
    if not filename.endswith(".ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as plyfile:

        # First magical word
        header = ["ply"]

        # Encoding format
        header.append("format binary_" + sys.byteorder + "_endian 1.0")

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # End of header
        header.append("end_header")

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, "ab") as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {"f": "float", "u": "uchar", "i": "int"}
    element = ["element " + name + " " + str(len(df))]

    if name == "face":
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append("property " + f + " " + df.columns.values[i])

    return element


def ply2dict(filename):
    data = read_ply(filename)
    dictionary = {}
    for name in data.dtype.names:
        dictionary[name] = data[name]
    return dictionary


def dict2ply(dictionary, filename):
    # fields = [f.reshape(-1, 1) for f in list(dictionary.values())]
    fields = list(dictionary.values())
    names = list(dictionary.keys())

    return write_ply(filename, fields, names)


##################################################################
#  Feature Descriptors
##################################################################

ORIENTATIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]

DESCRIPTORS = [
    "normals",
    "verticality",
    "linearity",
    "planarity",
    "sphericity",
    "curvature",
    "anisotropy",
    "surface_variation",
    ]


def local_PCA(points):

    eigenvalues = None
    eigenvectors = None

    n = points.shape[0]
    centroids = np.mean(points, axis=0)
    centered = points - centroids

    cov = centered.T @ centered / n

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues.astype(np.float32), eigenvectors.astype(np.float32)


def neighborhood_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all
    # query_points in cloud_points
    tree = KDTree(cloud_points)

    print("* querying radius ...")
    idx_lists = tree.query_radius(query_points, radius)

    all_eigenvalues = np.zeros((query_points.shape[0], 3), dtype=np.float32)
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3), dtype=np.float32)

    for i, idx_list in enumerate(tqdm(idx_lists, desc="* Processing neighborhoods")):
        if len(idx_list) > 0:
            points = cloud_points[idx_list]
            eigenvalues, eigenvectors = local_PCA(points)
            all_eigenvalues[i, :] = eigenvalues
            all_eigenvectors[i, :, :] = eigenvectors

    return all_eigenvalues, all_eigenvectors


def orient_normals(normals, preferred_orientation="+z"):
    index = ORIENTATIONS.index(preferred_orientation)
    sign = 1 if index % 2 == 0 else -1
    direction = index // 2
    normals[:, direction] = sign * np.abs(normals[:, direction])

    return normals


def compute_descriptors(coords, radius, descriptors, preferred_orientation, epsilon):

    if "all" in descriptors:
        descriptors = DESCRIPTORS
    assert len(descriptors) > 0
    assert np.all([d in DESCRIPTORS for d in descriptors])

    print(f"* descriptors: {descriptors}")
    print(f"* radius: {radius}")
    print(f"* epsilon: {epsilon}")
    print(f"* preferred normals orientation: {preferred_orientation}")
    # Compute the features for all points of the cloud
    eigenvalues, eigenvectors = neighborhood_PCA(coords, coords, radius)

    normals = orient_normals(eigenvectors[:, :, 0], preferred_orientation)

    normals_z = normals[:, 2]

    # L_1 >= L_2 >= L_3
    L_1 = eigenvalues[:, 2]
    L_2 = eigenvalues[:, 1]
    L_3 = eigenvalues[:, 0]

    # epsilon = 1e-2 * np.ones(len(normals_z))
    epsilon_array = epsilon * np.ones(len(normals_z), dtype=np.float32)
    all_descriptors = {}

    if "normals" in descriptors:
        all_descriptors["nx"] = normals[:, 0]
        all_descriptors["ny"] = normals[:, 1]
        all_descriptors["nz"] = normals[:, 2]

    if "verticality" in descriptors:
        verticality = 2 * np.arcsin(normals_z) / np.pi
        all_descriptors["verticality"] = verticality

    if "linearity" in descriptors:
        linearity = 1 - (L_2 / (L_1 + epsilon_array))
        all_descriptors["linearity"] = linearity

    if "planarity" in descriptors:
        planarity = (L_2 - L_3) / (L_1 + epsilon_array)
        all_descriptors["planarity"] = planarity

    if "sphericity" in descriptors:
        sphericity = L_3 / (L_1 + epsilon_array)
        all_descriptors["sphericity"] = sphericity

    if "curvature" in descriptors:
        curvature = L_3 / (L_1 + L_2 + L_3 + epsilon_array)
        all_descriptors["curvature"] = curvature

    if "anisotropy" in descriptors:
        anisotropy = (L_1 - L_3) / (L_1 + epsilon_array)
        all_descriptors["anisotropy"] = anisotropy

    if "surface_variation" in descriptors:
        surface_variation = L_3 / (L_1 + L_3 + L_3 + epsilon_array)
        all_descriptors["surface_variation"] = surface_variation

    return all_descriptors


##################################################################
#  Ground Extraction
##################################################################

METHODS_RASTERIZATION = ["closest_neighbor", "delaunay"]

def stitch_regions(coords, region_labels, slope_intra_max, slope_inter_max, percentile_closest):

    N = len(coords)
    ground_mask = np.zeros(N, dtype=bool)

    all_labels = np.unique(region_labels)
    all_labels = all_labels[all_labels > 0]

    spans = np.zeros(len(all_labels), dtype=np.float32)
    heights = np.zeros(len(all_labels), dtype=np.float32)

    # computing slopes intra
    for i, label in enumerate(tqdm(all_labels, desc="* Computing intra slopes")):
        coords_label = coords[region_labels == label]
        deltas = np.max(coords_label, axis=0) - np.min(coords_label, axis=0)
        spans[i] = np.linalg.norm(deltas[:2])
        heights[i] = deltas[2]

    slopes_intra = heights / spans

    # start with the region with the largest span
    init_label = all_labels[np.argmax(spans)]
    ground_mask[region_labels == init_label] = 1

    # computing slopes inter
    coords_ground = coords[ground_mask]

    tree = KDTree(coords_ground)
    for i, label in enumerate(tqdm(all_labels, desc="* Stitching regions together")):
        if label == init_label:
            continue

        if slopes_intra[i] > slope_intra_max:
            continue

        coords_label = coords[region_labels == label]
        dist, idx = tree.query(coords_label, k=1)
        idx = idx.ravel()
        dist = dist.ravel()

        mask_closest = dist < np.percentile(dist, 100 * percentile_closest)
        if np.sum(mask_closest) == 0:
            continue
        coords_label_closest = coords_label[mask_closest]

        idx_ground_closest = idx[mask_closest]
        coords_ground_closest = coords_ground[idx_ground_closest]

        deltas = np.abs(coords_label_closest - coords_ground_closest)

        dist_xy = np.linalg.norm(deltas[:, :2], axis=1)
        dist_z = deltas[:, 2]
        slopes_inter = dist_z / dist_xy

        mean_slopes_inter = np.mean(slopes_inter)

        if mean_slopes_inter < slope_inter_max:
            ground_mask[region_labels == label] = 1
            coords_ground = coords[ground_mask]
            tree = KDTree(coords_ground)

    return ground_mask


def interpolate_altitude(coords_ground, coords_queries_xy, method="delaunay"):

    if method == "closest_neighbor":
        # create a KD tree on xy coordinates
        tree = KDTree(coords_ground[:, :2])

        # find closest neighbor on the ground
        _, idx_neighbor = tree.query(coords_queries_xy, k=1)
        idx_neighbor = idx_neighbor.flatten()

        z_ground = coords_ground[:, -1]
        z_queries = z_ground[idx_neighbor]
        grid_3d = np.hstack((coords_queries_xy, z_queries.reshape(-1, 1)))

    elif method == "delaunay":
        # create 2D triangulation of ground coordinates
        tri = spatial.Delaunay(coords_ground[:, :2])

        # Find simplex of each query point
        idx_simplices = tri.find_simplex(coords_queries_xy)
        convex_hull_mask = idx_simplices >= 0

        # keep only query points inside convex hull
        idx_simplices = idx_simplices[convex_hull_mask]
        coords_queries_hull = coords_queries_xy[convex_hull_mask]

        # compute weights
        trans = tri.transform[idx_simplices]
        inv_T = trans[:, :-1, :]
        r = trans[:, -1, :]
        diff = (coords_queries_hull - r)[:, :, np.newaxis]
        barycent = (inv_T @ diff).squeeze()
        weights = np.c_[barycent, 1 - barycent.sum(axis=1)]

        # interpolate z values of vertices
        z_vertices = coords_ground[:, -1][tri.simplices][idx_simplices]
        z_queries = np.sum(weights * z_vertices, axis=1)
        grid_3d = np.hstack((coords_queries_hull, z_queries.reshape(-1, 1)))

    else:
        raise ValueError(f"Method '{method}' not found")

    return grid_3d


def rasterize_ground(coords, ground_mask, step_size, method):
    print(f"* method : {method}")
    print(f"* step_size : {step_size}")
    assert method in METHODS_RASTERIZATION

    mins = np.min(coords[:, :2], axis=0)
    maxs = np.max(coords[:, :2], axis=0)

    # Create a grid
    grid = np.mgrid[
        mins[0] : maxs[0] : step_size, mins[1] : maxs[1] : step_size
    ].T
    grid_points = grid.reshape(-1, 2)

    # Interpolate altitudes
    grid_3d = interpolate_altitude(coords[ground_mask], grid_points, method)

    return grid_3d


def height_above_ground(coords, ground_mask, grid_ground_3d):
    heights = np.zeros(len(coords), dtype=np.float32)

    coords_queries = coords[~ground_mask]

    tree = KDTree(grid_ground_3d[:, :2])

    # find closest neighbor on the rasterized ground
    _, idx_neighbor = tree.query(coords_queries[:, :2], k=1)
    idx_neighbor = idx_neighbor.flatten()

    # set heights
    z_ground = grid_ground_3d[:, -1]
    z_queries_ground = z_ground[idx_neighbor]
    heights[~ground_mask] = coords_queries[:, -1] - z_queries_ground

    return heights


##################################################################
#  Region Growing
##################################################################

def geometry_criterion(p1, p2, n1, n2, thresh_height, thresh_angle):

    criterion_height = (p2 - p1) @ n1 < thresh_height

    # dot product
    criterion_angle = np.abs(n2 @ n1) > np.cos(thresh_angle)
    # abs to avoid issues with normals in different directions

    return np.logical_and(criterion_height, criterion_angle)


def descriptor_criterion(descriptor, minimize, thresh_descriptor):
    if minimize:
        return descriptor < thresh_descriptor
    return descriptor > thresh_descriptor


def region_growing(coords, normals, radius, descriptor_vals, minimize, thresholds):

    N = len(coords)
    tree = KDTree(coords)
    region_mask = np.zeros(N, dtype=bool)

    Q_idx = [np.argmin(descriptor_vals) if minimize else np.argmax(descriptor_vals)]
    seen_idx = np.zeros(N, dtype=bool)
    i = 0

    while len(Q_idx) > 0:
        # print(f"  * N processed neighborhoods : {i}", end="\r")
        i += 1
        seed_idx = Q_idx.pop(0)

        seed_coords = coords[seed_idx]
        seed_normal = normals[seed_idx]

        neighbors_idx = tree.query_radius(seed_coords.reshape(1, -1), radius)
        neighbors_idx = neighbors_idx[0]

        # discard neighbors that have already been processed
        neighbors_idx = neighbors_idx[~seen_idx[neighbors_idx]]

        neighbors_points = coords[neighbors_idx]
        neighbors_normals = normals[neighbors_idx]

        # select neighbors
        geometry_mask = geometry_criterion(
            seed_coords,
            neighbors_points,
            seed_normal,
            neighbors_normals,
            thresholds["height"],
            thresholds["angle"],
        )
        selected_idx = neighbors_idx[geometry_mask]

        # add them to the region
        region_mask[selected_idx] = 1

        # add some of the to the queue
        selected_planarities = descriptor_vals[selected_idx]
        descriptor_mask = descriptor_criterion(selected_planarities, minimize, thresholds["descriptor"])

        queue_idx = selected_idx[descriptor_mask]

        # add processed indexes to the seen array
        seen_idx[neighbors_idx] = 1

        Q_idx += list(queue_idx)
        if i > 500000:
            print("Region growing stopped early : n_points > 500.000")
            break

    # print(f"  * Total number of points in region : {np.sum(region_mask)}")

    return region_mask


def multi_region_growing(coords, normals, descriptor_vals, radius, n_regions, minimize, thresholds):

    N = len(coords)
    is_region = np.zeros(N, dtype=bool)
    region_labels = -np.ones(N, dtype=np.int32)
    indexes = np.arange(N)

    for i in range(n_regions):
        # print(f"* Region {i + 1}/{n_regions}")
        label_i = i + 1
        region_mask = region_growing(coords, normals, radius, descriptor_vals, minimize, thresholds)

        idx_region = indexes[region_mask]
        region_labels[idx_region] = label_i
        is_region[idx_region] = 1

        coords = coords[~region_mask]
        normals = normals[~region_mask]
        descriptor_vals = descriptor_vals[~region_mask]

        indexes = indexes[~region_mask]

    n_regions_grown = len(np.unique(region_labels) - 1)
    print(f"* number of valid regions grown : {n_regions_grown}")

    return region_labels
