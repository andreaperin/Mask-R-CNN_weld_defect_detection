import torch
import torch.utils.data
import os
import torchvision
from datetime import datetime


from engine import get_model_frcnn_fpn_new_anchor, get_model_masck_fpn_new_anchor, train_one_epoch, evaluate, EarlyStopping
from dataset import get_transform,CERNDataset

import utils
from tensorboardX import SummaryWriter
from optparse import OptionParser

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage


parser = OptionParser()

##      Models settings
parser.add_option("--pt", action="store_true", dest="pretrained")
parser.add_option("--ete", action="store_false", dest="pretrained")

parser.add_option("--new_AS", action = 'store_true', dest='anchor_size')
parser.add_option("--std_AS", action = 'store_false', dest='anchor_size')

parser.add_option("--model", dest="model", type = int, help="choose 0 (mask) or 1 (frcnn)",
                  default = 0)

##      Dataset
parser.add_option('--dataset', dest = 'dataset', help = 'name of selected folder inside Datasets folder',
                  default = 'raccoon_dataset')
parser.add_option('--num_classes', dest = 'num_classes', type = int, help = 'number of classes + 1 (BG)',
                  default = 2)
parser.add_option('--workers', dest = 'num_workers', type = int, metavar = 'N',
                  default = 4)
parser.add_option('--epochs', dest = 'epochs', type = int, default = 10)
parser.add_option('--reduced', dest = 'reduced', type = int, help = 'select a value among [100; 70; 50; 30; 10; 5] as utilized dataset %')


##      Hyper-Parameters
parser.add_option('--batch_size', dest = 'batch_size', type = int, default = 1)
parser.add_option('--learning_rate', dest = 'learning_rate', type = float, metavar = 'LR', default = 5e-3)
parser.add_option('--momentum', dest = 'momentum', type = float, metavar = 'M', default = 0.9)
parser.add_option('--weight_decay', dest = 'weight_decay', type = float, default = 5e-4)

##      Early stopping
parser.add_option('--patience', dest = 'patience', default = 7, type = int)
parser.add_option('--delta', dest = 'delta', default = 0, type = int)

#       resume a non finisched training
parser.add_option('--resume', dest = 'resume', default='', type=str, metavar='PATH',
                  help='path to latest checkpoint (default: none)')



##      Directories
(options, args) = parser.parse_args()
dataset = options.dataset

if options.resume:
    resume = options.resume
    c = '/'
    position = [pos for pos, char in enumerate(resume) if char == c]
    exp_path = resume[:position[-2]]

else:
    #models_dir = os.path.join('/gpfs/work/pMI19_EneDa/aperin/tesi/pytorch_datasets','model')
    models_dir = os.path.join('model')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if dataset.find('/') > 0:
        dataset_id = dataset.replace('/', '_')

    else:
        dataset_id = dataset

    if options.pretrained == True:
        if not os.path.exists(os.path.join(models_dir,'Fine_Tuned_model')):
            os.mkdir(os.path.join(models_dir,'Fine_Tuned_model'))
        if options.reduced is None:
            super_folder = os.path.join(models_dir, 'Fine_Tuned_model', dataset_id)
        else:
            dataset_id_reduced = dataset_id + '_{}percent'.format(options.reduced)
            super_folder = os.path.join(models_dir, 'Fine_Tuned_model', dataset_id_reduced)

    else:
        if not os.path.exists(os.path.join(models_dir, 'end_to_end_model')):
            os.mkdir(os.path.join(models_dir, 'end_to_end_model'))
        if options.reduced is None:
            super_folder = os.path.join(models_dir, 'end_to_end_model', dataset_id)
        else:
            dataset_id_reduced = dataset_id + '_{}percent'.format(options.reduced)
            super_folder = os.path.join(models_dir, 'end_to_end_model', dataset_id)
    if not os.path.exists(super_folder):
        os.mkdir(super_folder)


    datetime_now = datetime.now().strftime('%b%d_%H-%M-%S')

    if options.model == 0:
        exp_name = 'mask_{}_AS.{}_{}epochs_'.format(dataset_id, options.anchor_size, options.epochs) + datetime_now
    if options.model == 1:
        exp_name = 'frcnn_{}_AS.{}_{}epochs_'.format(dataset_id, options.anchor_size, options.epochs) + datetime_now
    exp_path = os.path.join(super_folder, exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(os.path.join(exp_path))

tb_fold = 'tb_log'
tb_path = os.path.join(exp_path, tb_fold)
ckpt_fold = 'checkpoints'
ckpt_path = os.path.join(exp_path, ckpt_fold)
best_fold = 'best'
best_path = os.path.join(exp_path, best_fold)
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)
if not os.path.exists(tb_path):
    os.mkdir(tb_path)
if not os.path.exists(best_path):
    os.mkdir(best_path)


def main():
    dataset = options.dataset
    writer = SummaryWriter(tb_path)
    data_train = CERNDataset(root = os.path.join('Datasets', dataset),
                            data_file = os.path.join('Datasets', dataset,'data/labels.csv'),
                            transforms = get_transform(train = True))
    data_test = CERNDataset(root = os.path.join('Datasets', dataset),
                            data_file = os.path.join('Datasets', dataset,'data/labels.csv'),
                            transforms = get_transform(train = False))


    torch.manual_seed(1)

    indices = torch.randperm(len(data_train)).tolist()

    if options.dataset == 'GDXray_dataset/datasets_cropped':
        len_set = int((len(data_train)*0.4)*(options.reduced/100))
        splitter = int(len_set*0.3)
        dataset_train = torch.utils.data.Subset(data_train, indices[:(len_set - splitter)])
        dataset_test = torch.utils.data.Subset(data_test, indices[-splitter:])

        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=options.batch_size,
                                                       shuffle=True,
                                                       num_workers=options.num_workers,
                                                       collate_fn=utils.collate_fn)
        dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=options.num_workers,
                                                      collate_fn=utils.collate_fn)

    else:
        splitter = int(len(data_train) * 0.3)
        dataset_train = torch.utils.data.Subset(data_train, indices[:-splitter])
        dataset_test = torch.utils.data.Subset(data_test, indices[-splitter:])
        dataloader_train = torch.utils.data.DataLoader(dataset = dataset_train,
                                                       batch_size = options.batch_size,
                                                       shuffle = True,
                                                       num_workers = options.num_workers,
                                                       collate_fn = utils.collate_fn)
        dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test,
                                                      batch_size = 1,
                                                      shuffle = False,
                                                      num_workers = options.num_workers,
                                                      collate_fn = utils.collate_fn)
    # ## saving dataset in tb_log
    #
    # seq = iaa.Sequential([
    #     iaa.Resize({"height": 300, "width": 400})
    # ])
    #
    # arrays = []
    # images = []
    # arrays_aug_img = []
    # tensors_aug_img = []
    # labels = []
    # images_aug_with_boxes = []
    # tensors_aug_img_with_boxes = []
    # for batch_idx, (image, label) in enumerate(dataloader_train):
    #     img = image[0].numpy().transpose(1,2,0)
    #     img_aug = seq(image = img)
    #     img_aug_tensor = torch.from_numpy(img_aug.transpose(2,0,1))
    #
    #     images.append(image[0].unsqueeze(0))
    #     arrays.append(img)
    #     labels.append(label)
    #     arrays_aug_img.append(img_aug)
    #     tensors_aug_img.append(img_aug_tensor)
    #
    #     boxes = []
    #     for box in labels[0][0]['boxes']:
    #         box = box.tolist()
    #         boxes.append(box)
    #     my_bbs = []
    #     for b in boxes:
    #         bb = ia.BoundingBox(x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3])
    #         my_bbs.append(bb)
    #     bbs_oi = BoundingBoxesOnImage(my_bbs, shape = img.shape)
    #     img_aug, bbs_aug = seq(image = img, bounding_boxes = bbs_oi)
    #     bbs_aug_no_fout_clipart = bbs_aug.remove_out_of_image().clip_out_of_image()
    #     image_aug_with_boxes = bbs_aug_no_fout_clipart.draw_on_image(img_aug, size=2, color=[0,0,255])
    #     images_aug_with_boxes.append(image_aug_with_boxes)
    #     tensor_aug_img_with_boxes = torch.from_numpy(image_aug_with_boxes.transpose(2,0,1))
    #     tensors_aug_img_with_boxes.append(tensor_aug_img_with_boxes)
    #
    # grid = torchvision.utils.make_grid(tensors_aug_img, padding = 20)
    # grid_1 = torchvision.utils.make_grid(tensors_aug_img_with_boxes, padding = 20)
    # writer.add_image('images', grid)
    #
    # ##  TODO "Resize" imgaug-function seems to not work correctly in bboxes
    # writer.add_image('images_with_boxes', grid_1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = options.num_classes
    if options.model == 1:
        model = get_model_frcnn_fpn_new_anchor(num_classes, options.pretrained, options.anchor_size, False)
    if options.model == 0:
        model = get_model_masck_fpn_new_anchor(num_classes, options.pretrained, options.anchor_size)

    model.to(device)

    #writer.add_graph(model, tensors_aug_img)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params = params,
                                lr = options.learning_rate,
                                momentum = options.momentum,
                                weight_decay = options.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
                                                   step_size = 3,
                                                   gamma = 0.1)
    es = EarlyStopping(patience = options.patience,
                       verbose = True,
                       delta = options.delta)

    #   resume non finished training:
    if options.resume:
        if os.path.isfile(options.resume):
            print("loading weights '{}'".format(options.resume))
            checkpoint = torch.load(options.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            resume_epoch = checkpoint['epoch']
            print("loaded checkpoint'{}' (epoch {})".format(options.resume, checkpoint['epoch']))
        else:
            print("no checkpoint foung ar'{}'".format(options.resume))
    else:
        resume_epoch = 0

    num_epochs = options.epochs

    for epoch in range(resume_epoch, num_epochs):
        train_one_epoch(model = model,
                        optimizer = optimizer,
                        data_loader = dataloader_train,
                        device = device,
                        epoch = epoch,
                        print_freq = 10,
                        writer = writer,
                        ckpt_path = ckpt_path)

        lr_scheduler.step()
        (coco, map05) = evaluate(model = model,
                                 data_loader = dataloader_test,
                                 device = device,
                                 writer = writer,
                                 epoch = epoch)
        es(val_acc = map05, model = model, path = best_path, epoch = epoch, optimizer = optimizer)

if __name__ == main():
    main()
