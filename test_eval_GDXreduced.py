import numpy as np
import torch.utils.data
from PIL import Image
import os

import utils
from PIL import ImageDraw, Image
import engine_std
from engine import get_model_frcnn_fpn_new_anchor, get_model_masck_fpn_new_anchor, train_one_epoch, evaluate, EarlyStopping
from dataset import get_transform, CERNDataset_Test


cnn = input("Enter 70, 50, 30, 10 for the % of GDX dataset:")
if cnn == '70':
    cnn_1 = 'mask_70_GDX'
if cnn == '50':
    cnn_1 = 'mask_50_GDX'
if cnn == '30':
    cnn_1 = 'mask_30_GDX'
if cnn == '10':
    cnn_1 = 'mask_10_GDX'

score_thresh = input("Enter a score thresh (0;1):")
gen_path = '/media/pero/pero_hdd/Thesis/models/last_models/model/Fine_Tuned_model/GDXray_reduced'
mask_70 ='GDXray_dataset_datasets_cropped_70percent/mask_GDXray_dataset_datasets_cropped_AS.False_90epochs_Sep03_13-21-56/best/ckpt_epoch_24.pth'
mask_50 ='GDXray_dataset_datasets_cropped_50percent/mask_GDXray_dataset_datasets_cropped_AS.False_90epochs_Sep03_13-18-00/best/ckpt_epoch_33.pth'
mask_30 ='GDXray_dataset_datasets_cropped_30percent/mask_GDXray_dataset_datasets_cropped_AS.False_90epochs_Sep03_13-22-57/best/ckpt_epoch_56.pth'
mask_10 ='GDXray_dataset_datasets_cropped_10percent/mask_GDXray_dataset_datasets_cropped_AS.False_90epochs_Sep03_13-23-11/best/ckpt_epoch_10.pth'
dataset = '/media/pero/pero_hdd/Thesis/pytorch_frcnn_mask/Datasets/GDXray_dataset/datasets_cropped'

exp_path = 'test/GDXray_reduced'
if not os.path.exists(exp_path):
    os.mkdir(exp_path)
'''
    Foldering
                '''
if not os.path.exists(os.path.join(exp_path, cnn_1)):
    os.mkdir(os.path.join(exp_path, cnn_1))
test_path = os.path.join(exp_path, cnn_1, str(score_thresh))
if not os.path.exists(test_path):
    os.mkdir(test_path)

'''
    Model Loading
                    '''

if cnn == '70':
    model_to_test = get_model_masck_fpn_new_anchor(num_classes = 2, pretrained= True, new_AS = False)
    ckpt = torch.load(os.path.join(gen_path, mask_70))
    model_to_test.load_state_dict(ckpt['model'])


if cnn == '50':
    model_to_test = get_model_masck_fpn_new_anchor(num_classes = 2, pretrained= True, new_AS = False)
    ckpt = torch.load(os.path.join(gen_path, mask_50))
    model_to_test.load_state_dict(ckpt['model'])

if cnn == '30':
    model_to_test = get_model_masck_fpn_new_anchor(num_classes = 2, pretrained = True, new_AS = False)
    ckpt = torch.load(os.path.join(gen_path, mask_30))
    model_to_test.load_state_dict(ckpt['model'])

if cnn == '10':
    model_to_test = get_model_masck_fpn_new_anchor(num_classes = 2, pretrained = True, new_AS = False)
    ckpt = torch.load(os.path.join(gen_path, mask_10))
    model_to_test.load_state_dict(ckpt['model'])

dataset_test = CERNDataset_Test(root = os.path.join(dataset, 'test'),
                                data_file = os.path.join(dataset, 'test', 'data/labels.csv'),
                                transforms = get_transform(train = False))

''''
    images processing 
                        '''

print('images processing has started')

for idx in range(len(dataset_test.imgs)):
    img, _ = dataset_test[idx]
    label_boxes = np.array(dataset_test[idx][1]["boxes"])

    model_to_test.eval()
    with torch.no_grad():
        prediction = model_to_test([img])

    image = Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
    draw = ImageDraw.Draw(image)

    if os.path.exists(os.path.join(dataset, 'test', 'data/label.csv')):
        # GT
        for elem in range(len(label_boxes)):
            draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                            (label_boxes[elem][2], label_boxes[elem][3])],
                           outline = 'green', width = 1)
        # Predictions
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().detach().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().detach().numpy(), decimals = 3)
            if score > float(score_thresh):
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                               outline ="red", width =1)
                draw.text((boxes[0], boxes[1]), text = str(score))

        image.save(os.path.join(test_path, dataset_test.imgs[idx]))

    else:
        # Predictions
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().detach().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().detach().numpy(), decimals = 3)
            if score > float(score_thresh):
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                               outline ="red", width =3)
                draw.text((boxes[0], boxes[1]), text = str(score))

        image.save(os.path.join(test_path, dataset_test.imgs[idx]))

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

coco_evaluator = engine_std.evaluate(model = model_to_test,
                                     data_loader = dataloader_test,
                                     device = torch.device('cpu'))

result_file_path = os.path.join(exp_path, cnn_1, 'result_file_score{}.txt'.format(float(score_thresh)))
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

