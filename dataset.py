import pandas as pd
import torch
import os
from PIL import Image
import transforms as T

''' 
    In order to be used datased should be constructed as follow:
        Dataset:
            - annotations (.xml file with annotations)
            - data (.cvs file)
            - images (.jpg file)
    '''

def parse_one_anno(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    return boxes_array

class CERNDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms = None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_anno(self.path_to_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype = torch.float32)

        num_objs = len(box_list)
        labels = torch.ones((num_objs,), dtype = torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append((T.RandomHorizontalFlip(0.5)))
    return T.Compose(transforms)

''''
        TEST    '''

class CERNDataset_Test(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms = None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        if self.path_to_data_file is not None:
            img_path = os.path.join(self.root, "images", self.imgs[idx])
            img = Image.open(img_path).convert("RGB")
            box_list = parse_one_anno(self.path_to_data_file, self.imgs[idx])
            boxes = torch.as_tensor(box_list, dtype = torch.float32)

            num_objs = len(box_list)
            labels = torch.ones((num_objs,), dtype = torch.int64)
            image_id = torch.tensor([idx])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)

        if self.path_to_data_file is None:
            img_path = os.path.join(self.root, "images", self.imgs[idx])
            img = Image.open(img_path).convert("RGB")

            if self.transforms is not None:
                target = None
                img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
