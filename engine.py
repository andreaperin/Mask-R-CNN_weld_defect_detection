import math
import sys
import time
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.detection.mask_rcnn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection import roi_heads

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def get_model_frcnn_fpn_new_anchor(num_classes, pretrained, new_AS, focal_loss):
    if pretrained == True:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    else:
        model =torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)

    if new_AS == True:
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

    # SET CLASSES NUMEBR
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    if focal_loss == True:
        roi_heads.fastrcnn_loss = focal_loss

    return model

def get_model_masck_fpn_new_anchor(num_classes, pretrained, new_AS):
    if pretrained == True:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = False)

    if new_AS == True:
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256,512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        # CHANGE ANCHOR SIZES
        model.rpn.anchor_generator = anchor_generator
        anchor_generator = AnchorGenerator(
            sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        model.rpn.anchor_generator = anchor_generator

        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        model.roi_heads.mask_roi_pool = None
    else:
        model.roi_heads.mask_roi_pool = None
    # SET CLASSES NUMEBR
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    return model

def set_trainable_layer(model, layers):

    ## all layers (before)
    for name, param in model.named_parameters():
        print(name, ':', param.requires_grad)

    ## all modules
    for name, child in model.named_children():
        print('name: ', name)
        print('isinstance({}, nn.Module): '.format(name), isinstance(child, nn.Module))
        print('=====')
        print(child)

    ## setting trainable layers:
        layers = 20
        ct = 0
        for name, param in model.named_parameters():
            ct +=1
            if ct < layers:
                param.requires_grad == False
                print('layer: {} is set to not trainable'.format(name))


class EarlyStopping:
    """
            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement.
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model, path, epoch, optimizer):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_best(model, path, epoch, optimizer)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best(model, path, epoch, optimizer)
            self.counter = 0

    def save_best(self, model, path, epoch, optimizer):
        best_dict = {'epoch': epoch+1,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

        a = os.listdir(path)
        for i in a:
            os.remove(os.path.join(path, i))

        torch.save(best_dict,
                   os.path.join(path, 'ckpt_epoch_' + str(epoch + 1) + '.pth'))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer, ckpt_path):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            writer.add_scalar('Training Loss', loss_value, epoch * len(data_loader) + batch_idx)
            writer.add_scalar('loss_classifier', loss_dict_reduced['loss_classifier'].item(),
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar('loss_box_reg', loss_dict_reduced['loss_box_reg'].item(),
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar('loss_objectness', loss_dict_reduced['loss_objectness'].item(),
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar('loss_rpn_box_reg', loss_dict_reduced['loss_rpn_box_reg'].item(),
                              epoch * len(data_loader) + batch_idx)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).cpu().item()
                    writer.add_histogram(name + '_grad', param_norm, epoch)
                # else:
                #     print("{} has no grad".format(name))

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    #   Save model
    print("Saving model at training epoch: {}".format(epoch + 1))
    ckpt_dict = {'epoch': epoch+1,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
    torch.save(ckpt_dict,
               os.path.join(ckpt_path, 'ckpt_epoch-' + str(epoch + 1) + 'loss' + str(loss_value) + '.pth'))




def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    ## TODO commeted cause I am using mask without mask block
    # if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
    #     iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, writer, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    #   TB Logs (remember, when not specified MaxDet = 100)

    writer.add_scalar('mAP @ [.5:.0.5:.95] @ all_size', coco_evaluator.coco_eval['bbox'].stats[0], epoch)
    writer.add_scalar('mAP @ 0.5 @ all_size', coco_evaluator.coco_eval['bbox'].stats[1], epoch)
    writer.add_scalar('mAP @ 0.75 @ all_size', coco_evaluator.coco_eval['bbox'].stats[2], epoch)
    writer.add_scalar('mAP @ [.5:.0.5:.95] @ SMALL', coco_evaluator.coco_eval['bbox'].stats[3], epoch)
    writer.add_scalar('mAP @ [.5:.0.5:.95] @ MEDIUM', coco_evaluator.coco_eval['bbox'].stats[4], epoch)
    writer.add_scalar('mAP @ [.5:.0.5:.95] @ LARGE', coco_evaluator.coco_eval['bbox'].stats[5], epoch)

    writer.add_scalar('mAR @ [.5:.0.5:.95] @ all_size @ MaxDet=1', coco_evaluator.coco_eval['bbox'].stats[6], epoch)
    writer.add_scalar('mAR @ [.5:.0.5:.95] @ all_size @ MaxDet=10', coco_evaluator.coco_eval['bbox'].stats[7], epoch)
    writer.add_scalar('mAR @ [.5:.0.5:.95] @ all_size @ MaxDet=100', coco_evaluator.coco_eval['bbox'].stats[8], epoch)
    writer.add_scalar('mAR @ [.5:.0.5:.95] @ SMALL', coco_evaluator.coco_eval['bbox'].stats[9], epoch)
    writer.add_scalar('mAR @ [.5:.0.5:.95] @ MEDIUM', coco_evaluator.coco_eval['bbox'].stats[10], epoch)
    writer.add_scalar('mAR @ [.5:.0.5:.95] @ LARGE', coco_evaluator.coco_eval['bbox'].stats[11], epoch)

    return coco_evaluator, coco_evaluator.coco_eval['bbox'].stats[1]

