import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import os, imageio
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from engine_std import evaluate
import utils
from PIL import ImageDraw
import transforms as T

def parse_one_anno(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    return boxes_array

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

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append((T.RandomHorizontalFlip(0.5)))
    return T.Compose(transforms)


path_to_external_GDX = '/home/pero/PycharmProjects/pytorch_frcnn_mask/Datasets/GDXray_dataset/datasets_cropped/test'
path_to_external_CERN = '/home/pero/PycharmProjects/pytorch_frcnn_mask/Datasets/CERN_defects_new_split_dataset/test'

def GT_drawer(test_dataset_path):
    data_file = os.path.join(test_dataset_path, 'data/labels.csv')
    GT_folder_path = os.path.join(test_dataset_path, 'GT_images')
    if not os.path.exists(GT_folder_path):
        os.mkdir(GT_folder_path)
    dataset_test = CERNDataset_Test(root = test_dataset_path,
                                    data_file = data_file,
                                    transforms = get_transform(train=False))
    for idx in range(len(dataset_test.imgs)):
        img, _ = dataset_test[idx]
        label_boxes = np.array(dataset_test[idx][1]["boxes"])


        image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        draw = ImageDraw.Draw(image)
        for elem in range(len(label_boxes)):
            draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                            (label_boxes[elem][2], label_boxes[elem][3])],
                           outline='green', width=1)

        image.save(os.path.join(GT_folder_path, dataset_test.imgs[idx]))



