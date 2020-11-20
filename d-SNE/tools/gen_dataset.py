import os
import argparse
import numpy as np
import struct
import pickle as pk
import h5py


class Dataset:
    def __init__(self, root):
        self.root = root

    def read(self):
        pass
    
    def dump(self):
        with open(os.path.join(self.root, self.__class__.__name__ + '.pkl'), 'wb') as fout:
            pk.dump(self.dataset, fout)



class MNIST(Dataset):
    def read(self):
        def read_img(path):
            with open(path, 'rb') as fimg:
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                img = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows, cols)

            return img

        def read_lbl(path):
            with open(path, 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)

            return lbl

        train_img = read_img(os.path.join(self.root, 'train-images-idx3-ubyte'))
        train_lbl = read_lbl(os.path.join(self.root, 'train-labels-idx1-ubyte'))

        test_img = read_img(os.path.join(self.root, 't10k-images-idx3-ubyte'))
        test_lbl = read_lbl(os.path.join(self.root, 't10k-labels-idx1-ubyte'))
        
        self.dataset = {'TR': [train_img, train_lbl], 'TE': [test_img, test_lbl]}
        

class MNISTM(Dataset):
    def read(self):
        def read_lbl(path):
            with open(path, 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)
                
        file_path = os.path.join(self.root, "keras_mnistm.pkl")
        with open(file_path, "rb") as f:
            mnist_m_data = pk.load(f, encoding="bytes")
        train = mnist_m_data[b"train"]
        test = mnist_m_data[b"test"]

        train_lbl = read_lbl(os.path.join(self.root, 'train-labels-idx1-ubyte'))
        test_lbl = read_lbl(os.path.join(self.root, 't10k-labels-idx1-ubyte'))
        
        self.dataset = {'TR': [train, train_lbl], 'TE': [test, test_lbl]}
        
class USPS(Dataset):
    def read(self):
        path = os.path.join(self.root, "usps.h5")
        with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:].reshape(-1, 16, 16)
            y_tr = train.get('target')[:]
            
            test = hf.get('test')
            X_te = test.get('data')[:].reshape(-1, 16, 16)
            y_te = test.get('target')[:]
            
        self.dataset = {'TR': [X_tr, y_tr], 'TE': [X_te, y_te]}


dataset_dict = {
    "mnist": MNIST,
    "mnistm": MNISTM,
    "usps": USPS,
}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', )
    parser.add_argument('-d', '--dataset', default='mnist', help='dataset')

    args = parser.parse_args()

    if args.dataset.lower() in dataset_dict:
        dataset = dataset_dict[args.dataset.lower()](args.dir)
    else:
        print('Required download the dataset and packed by yourself, sorry for inconvenience')
        
    dataset.read()
    dataset.dump()
