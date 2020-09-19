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


def get_model_frcnn_test(num_classes, new_as):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    if new_as == True:
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        # CHANGE ANCHOR SIZES
        model.rpn.anchor_generator = anchor_generator
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        model.rpn.anchor_generator = anchor_generator

        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    return model

def get_model_mask_test(num_classes, new_as):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    if new_as == True:
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        # CHANGE ANCHOR SIZES
        model.rpn.anchor_generator = anchor_generator
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        model.rpn.anchor_generator = anchor_generator

        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    model.roi_heads.mask_roi_pool = None
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append((T.RandomHorizontalFlip(0.5)))
    return T.Compose(transforms)


model_path = 'mask_CERN_defects_dataset_AS.True_90epochs_Jun20_13-00-26/checkpoints/ckpt_epoch-70loss0.2819882333278656.pth'
dataset = 'CERN_defects_new_split_dataset'


path_to_external_GDX ='/media/pero/TOSHIBA EXT/Thesis/models/new_frcnn-mask/model/Fine_Tuned_model/GDXray_dataset_datasets_cropped'
path_to_external_CERN = '/media/pero/TOSHIBA EXT/Thesis/models/new_frcnn-mask/model/Fine_Tuned_model/CERN_defects_dataset'
if dataset.__contains__('CERN'):
    ckpt_path = os.path.join(path_to_external_CERN, model_path)
if dataset.__contains__('GDX'):
    ckpt_path = os.path.join(path_to_external_GDX, model_path)

c = '/'
position = [pos for pos, char in enumerate(ckpt_path) if char == c]
exp_path = ckpt_path[:position[-2]]
if not os.path.exists(os.path.join(exp_path, 'test')):
    os.mkdir(os.path.join(exp_path, 'test'))
test_path_low_score = os.path.join(exp_path, 'test', 'score_thresh_0.3')
test_path_high_score = os.path.join(exp_path, 'test', 'score_thresh_0.8')
if not os.path.exists(test_path_low_score):
    os.mkdir(test_path_low_score)
if not os.path.exists(test_path_high_score):
    os.mkdir(test_path_high_score)

if os.path.exists(os.path.join('Datasets', dataset, 'test/data/labels.csv')):
    data_file = os.path.join('Datasets', dataset, 'test/data/labels.csv')
else:
    data_file = None


if model_path.__contains__('AS.True'):
    new_as = True
if model_path.__contains__('AS.False'):
    new_as = False

if model_path.__contains__('frcnn'):
    loaded_model = get_model_frcnn_test(num_classes=2, new_as=new_as)
    ckpt = torch.load(ckpt_path)
    loaded_model.load_state_dict(ckpt['model'])

if model_path.__contains__('mask'):
    loaded_model = get_model_mask_test(num_classes=2, new_as=new_as)
    ckpt = torch.load(ckpt_path)
    loaded_model.load_state_dict(ckpt['model'])


dataset_test = CERNDataset_Test(root = os.path.join('Datasets', dataset, 'test'),
                                data_file = data_file,
                                transforms=get_transform(train=False))

''''
    images processing 
                        '''

print('images processing has started')

for idx in range(len(dataset_test.imgs)):
    img, _ = dataset_test[idx]
    label_boxes = np.array(dataset_test[idx][1]["boxes"])

    loaded_model.eval()
    with torch.no_grad():
        prediction = loaded_model([img])

    image = Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
    draw = ImageDraw.Draw(image)

    if os.path.exists(os.path.join('Datasets', dataset, 'test/data/labels.csv')):

        # GT
        for elem in range(len(label_boxes)):
            draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                            (label_boxes[elem][2], label_boxes[elem][3])],
                           outline = 'green', width = 1)
        # Predictions
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().detach().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().detach().numpy(), decimals = 3)
            if score > 0.3:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                               outline ="red", width =1)
                draw.text((boxes[0], boxes[1]), text = str(score))

        image.save(os.path.join(test_path_low_score,dataset_test.imgs[idx]))
        for element in range(len(prediction[0]["boxes"])):
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                           outline="red", width=3)
                draw.text((boxes[0], boxes[1]), text=str(score))

        image.save(os.path.join(test_path_high_score, dataset_test.imgs[idx]))

    else:
        # Predictions
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().detach().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().detach().numpy(), decimals = 3)
            if score > 0.3:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                               outline ="red", width =3)
                draw.text((boxes[0], boxes[1]), text = str(score))

        image.show(test_path_low_score)

        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().detach().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().detach().numpy(), decimals = 3)
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                               outline ="red", width =3)
                draw.text((boxes[0], boxes[1]), text = str(score))

        image.show(test_path_high_score)

print("images processing is over")

'''
    Network test performance evaluation
                                        '''

print('Network performance evaluation on test set is started')

dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test,
                                              batch_size = 1,
                                              shuffle = False,
                                              num_workers = 4,
                                              collate_fn = utils.collate_fn)

coco_evaluator = evaluate(model = loaded_model,
                          data_loader = dataloader_test,
                          device = torch.device('cpu'))

result_file_path = os.path.join(exp_path, 'test', 'result_file.txt')
with open(result_file_path, 'w') as f:
    f.write('mAP @ [.5:.0.5:.95] @ all_size  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[0]))
    f.write('mAP @ 0.5 @ all_size  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[1]))
    f.write('mAP @ 0.75 @ all_size  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[2]))
    f.write('mAP @ [.5:.0.5:.95] @ SMALL  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[3]))
    f.write('mAP @ [.5:.0.5:.95] @ MEDIUM  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[4]))
    f.write('mAP @ [.5:.0.5:.95] @ LARGE  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[5]))
    f.write('mAR @ [.5:.0.5:.95] @ all_size @ MaxDet=1  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[6]))
    f.write('mAR @ [.5:.0.5:.95] @ all_size @ MaxDet=10  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[7]))
    f.write('mAR @ [.5:.0.5:.95] @ all_size @ MaxDet=100  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[8]))
    f.write('mAR @ [.5:.0.5:.95] @ SMALL  {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[9]))
    f.write('mAR @ [.5:.0.5:.95] @ MEDIUM {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[10]))
    f.write('mAR @ [.5:.0.5:.95] @ LARGE {}\r\n'.format(coco_evaluator.coco_eval['bbox'].stats[11]))






