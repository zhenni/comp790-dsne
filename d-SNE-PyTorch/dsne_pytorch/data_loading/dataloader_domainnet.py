import os
import os.path
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(file_name, file_type):
    import pdb
    pdb.set_trace()
    
    base_path = './data/domainnet'
    root = './data/domainnet'
    image_set_file = os.path.join(base_path, file_name)

    crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            ResizeImage(256),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dataset = Imagelists_VISDA(image_set_file, root=root,
                                      transform=data_transforms[file_type])

    class_list = return_classlist(image_set_file)
    print("%d classes in this dataset" % len(class_list))
    bs = 16
    if file_type == 'train':
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                                num_workers=2, shuffle=True,
                                                drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=min(bs, len(dataset)),
                                    num_workers=2,
                                    shuffle=True, drop_last=True)

    return data_loader, class_list


def domainnet(root_path, filter_num_cls=1, split='train'):
    paths, labels = make_dataset_fromlist(root_path + "_{}.txt".format(split))
    
    idx = labels < filter_num_cls # filter
    labels = labels[idx]
    labels = np.array(labels, dtype=np.uint8)

    paths = paths[idx]
    images = np.array([np.array(cv2.cvtColor(cv2.imread(os.path.join(root_path, path)), cv2.COLOR_BGR2RGB), dtype=np.uint8) for path in paths], dtype=object)
    return images, labels


if __name__ == "__main__":
    # execute only if run as a script
    return_dataset(file_name="clipart_train.txt", file_type="train")