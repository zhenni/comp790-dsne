"""Script for packing downloaded datasets into HDF5 container.

Arguments
---------
datasets : str or list of str
    Names of datasets to try to pack into HDF5 containers. Must be
    present in the following list: MNIST, MNIST-M
"""

# Stdlib imports
import argparse
import os
from pathlib import Path
import gzip
import struct

# Third-party imports
import h5py
import numpy as np
import cv2
import torchvision


def mnist(root_path):
    """Load serialized MNIST labels and images.

    Parameters
    ----------
    root_path : str or Path object
        Path to root directory containing image/label archives.

    Returns
    -------
    dataset : dict of ndarrays
        Dictionary containing four items: training labels `y_tr`,
        training images `X_tr`, testing labels `y_te`, and testing
        images `X_te`.

    Notes
    -----
    MNIST dataset must be downloaded manually. A mirror can be found at
    the following location as of 2020-03-31:

    http://yann.lecun.com/exdb/mnist/
    """
    # Set up paths for required directory and files
    mnist_path = Path(root_path)
    required_files = {"y_tr": "train-labels-idx1-ubyte.gz",
                          "X_tr": "train-images-idx3-ubyte.gz",
                      "y_te": "t10k-labels-idx1-ubyte.gz",
                      "X_te": "t10k-images-idx3-ubyte.gz"}

    # Check to see if required files exist
    assert mnist_path.exists()
    for file_path in required_files.values():
        assert (mnist_path / file_path).exists()

    # Load labels and images into dataset dictionary
    dataset = {}
    for name, file_path in required_files.items():
        with gzip.open(mnist_path / file_path, 'rb') as f:
            if name.startswith("y"):
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)

            elif name.startswith("X"):
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8)
                data = data.reshape(-1, rows, cols)
                data = np.stack((data, )*3, axis=-1)  # Grayscale -> 3-channel

            dataset[name] = data

    return dataset


def mnist_m(root_path):
    """Load serialized MNIST-M labels and images.

    Parameters
    ----------
    root_path : str or Path object
        Path to root directory containing image/label archives.

    Returns
    -------
    dataset : dict of ndarrays
        Dictionary containing four items: training labels `y_tr`,
        training images `X_tr`, testing labels `y_te`, and testing
        images `X_te`.

    Notes
    -----
    MNIST-M dataset must be downloaded manually. A mirror can be found
    at the following location as of 2020-03-31:

    https://github.com/fungtion/DANN
    """
    # Set up paths for required directory and files
    mnist_m_path = Path(root_path)
    required_files = {"y_tr": "mnist_m_train_labels.txt",
                      "X_tr": "mnist_m_train",
                      "y_te": "mnist_m_test_labels.txt",
                      "X_te": "mnist_m_test"}

    # Check to see if required files exist
    assert mnist_m_path.exists()
    for file_path in required_files.values():
        assert (mnist_m_path / file_path).exists()

    # Load labels and images into dataset dictionary
    dataset = {}
    for name, file_path in required_files.items():
        if name.startswith('y'):
            with open(root_path/file_path, 'r') as f:
                paths, labels = zip(*[line.split() for line in f.readlines()])
                dataset[name] = np.array(labels, dtype=np.uint8)

        elif name.startswith("X"):
            images = [cv2.cvtColor(cv2.imread(str(root_path/file_path/path)),
                                   cv2.COLOR_BGR2RGB)
                      for path in paths]
            dataset[name] = np.array(images, dtype=np.uint8)

    return dataset


def svhn(root_path):
    svhn_path = Path(root_path)
    train = torchvision.datasets.SVHN(root=svhn_path, download=True, split='train')
    test = torchvision.datasets.SVHN(root=svhn_path, download=True, split='test')
    dataset = {}
    y_tr = np.zeros(len(train), dtype=np.uint8)
    X_tr = np.zeros((len(train), *(train[0][0].size), 3), dtype=np.uint8)
    for idx, (xx, yy) in enumerate(train):
                    y_tr[idx] = yy
                    X_tr[idx] = np.asarray(xx)
    dataset['y_tr'] = y_tr
    dataset['X_tr'] = X_tr
    y_te = np.zeros(len(test), dtype=np.uint8)
    X_te = np.zeros((len(test), *(test[0][0].size), 3), dtype=np.uint8)
    for idx, (xx, yy) in enumerate(test):
                    y_te[idx] = yy
                    X_te[idx] = np.asarray(xx)
    dataset['y_te'] = y_te
    dataset['X_te'] = X_te
    
    return dataset
         
    
def usps(root_path):
#     usps_path = Path(root_path)
#     path = os.path.join(usps_path, "usps.h5")
#     with h5py.File(path, 'r') as hf:
#         train = hf.get('train')
                
#         X_tr = train.get('data')[:].reshape(-1, 16, 16)
#         X_tr = np.stack((X_tr, )*3, axis=-1)
#         y_tr = train.get('target')[:]

#         test = hf.get('test')
#         X_te = test.get('data')[:].reshape(-1, 16, 16)
#         X_te = np.stack((X_te, )*3, axis=-1)
#         y_te = test.get('target')[:]
    
#     dataset = {}
#     dataset['X_tr'] = X_tr
#     dataset['y_tr'] = y_tr
#     dataset['X_te'] = X_te
#     dataset['y_te'] = y_te

    
    usps_path = Path(root_path)
    train = torchvision.datasets.USPS(root=usps_path, download=True, train=True)
    test = torchvision.datasets.USPS(root=usps_path, download=True, train=False)
    dataset = {}
    y_tr = np.zeros(len(train), dtype=np.uint8)
    X_tr = np.zeros((len(train), 16, 16, 3), dtype=np.uint8)

    for idx, (xx, yy) in enumerate(train):
                    y_tr[idx] = yy
                    data = np.asarray(xx).reshape(-1, 16, 16)
                    data = np.stack((data, )*3, axis=-1)  # Grayscale -> 3-channel
                    X_tr[idx] = data
    dataset['y_tr'] = y_tr
    dataset['X_tr'] = X_tr
    y_te = np.zeros(len(test), dtype=np.uint8)
    X_te = np.zeros((len(test), 16, 16, 3), dtype=np.uint8)
    for idx, (xx, yy) in enumerate(test):
                    y_te[idx] = yy
                    data = np.asarray(xx).reshape(-1, 16, 16)
                    data = np.stack((data, )*3, axis=-1)  # Grayscale -> 3-channel
                    X_te[idx] = data
    dataset['y_te'] = y_te
    dataset['X_te'] = X_te

    return dataset


def pack_dataset(output_path, dataset):
    """Pack image dataset into HDF5 container.

    Parameters
    ----------
    output_path : str or Path object
        Path representing HDF5 file to pack dataset into.
    dataset : dict of ndarrays
        Dictionary containing four items: training labels `y_tr`,
        training images `X_tr`, testing labels `y_te`, and testing
        images `X_te`.
    """
    output_path = Path(output_path)
    if output_path.exists():
        renamed_path = f"{output_path.stem}_backup{output_path.suffix}"
        print(f"{output_path} already exists. Renaming to {renamed_path}.")
        output_path.rename(renamed_path)

    file = h5py.File(output_path, "w")
    
    for split_name in ["tr", "te"]:
        image_name = f"X_{split_name}"
        label_name = f"y_{split_name}"

        images = dataset[image_name]
        labels = dataset[label_name]

        file.create_dataset(image_name, np.shape(images), h5py.h5t.STD_U8BE,
                            data=images)
        file.create_dataset(label_name, np.shape(labels), h5py.h5t.STD_U8BE,
                            data=labels)

    file.close()


def main(requested_datasets):
    # Ensure that current working directory is where pack_data_hdf5.py is
    os.chdir(Path(os.path.realpath(__file__)).parent)

    # Mapping from str arguments to function names
    dataset_funcs = {"mt": mnist, "mnist": mnist,
                     "mm": mnist_m, "mnistm": mnist_m, "mnist-m": mnist_m,
                     "s": svhn, "svhn": svhn,
                     "u": usps, "usps": usps,}

    for dataset in requested_datasets:
        try:
            func = dataset_funcs[dataset.lower()]

            # Use function name as convention for directory/filenames
            dataset_path = Path("data") / func.__name__
            output_path = Path("data") / f"{func.__name__}.h5"

            dataset = func(dataset_path)
            pack_dataset(output_path, dataset)

        except KeyError:
            print(f'"{dataset}" does not map to valid dataset. Skipping it.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    main(args.datasets)
