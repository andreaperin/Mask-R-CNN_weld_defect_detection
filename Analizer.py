from PIL import Image
import pandas as pd
import os, imageio
import xml.etree.ElementTree as ET
import torch
from PIL import ImageDraw


def parse_one_anno(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    return boxes_array

dir = 'Datasets/GDXray_dataset/datasets_cropped'
img_list = os.listdir(os.path.join(dir, 'images'))
targets = {}
for img in img_list:
    boxes = []
    areas = []
    targets[img] = {}
    image = Image.open(os.path.join(dir, 'images', img))
    box_id_i = img[:-3] +'xml'
    box_id = os.path.join(dir, 'annotations',box_id_i)
    tree = ET.parse(box_id)
    root = tree.getroot()
    for object in root.iter('object'):
        for bndbox in object.iter('bndbox'):
            for x_min in bndbox.iter('xmin'):
                x1 = float(x_min.text)
            for x_max in bndbox.iter('xmax'):
                x2 = float(x_max.text)
            for y_min in bndbox.iter('ymin'):
                y1 = float(y_min.text)
            for y_max in bndbox.iter('ymax'):
                y2 = float(y_max.text)
            boxes.append([x1, y1, x2, y2])
            area = (x2-x1)*(y2-y1)
            areas.append(area)
            if area == 0:
                root.remove(object)
    tree.write(box_id)
    targets[img]['boxes'] = boxes
    targets[img]['areas'] = areas


'''
removing images with no bboxex, and bboxes_W0001 with area == 0
    '''
too_low_areas = []
img_low_area = []
for key in targets:
    if len(targets[key]['boxes']) == 0:
        os.remove(os.path.join('Datasets/GDXray_dataset/datasets_cropped/images', key))
        os.remove(os.path.join('Datasets/GDXray_dataset/datasets_cropped/annotations', key[:-3]+'xml'))
    for are in targets[key]['areas']:
        if are == 0:
            #print('{}_too low area'. format(are))
            too_low_areas.append(are)
            img_low_area.append(key)

'''
check on bbounding boxes showing some images
    '''
heights = []
widths = []
for img in os.listdir('Datasets/GDXray_dataset/datasets_cropped/images'):
    boxes = []
    image = Image.open(os.path.join('Datasets/GDXray_dataset/datasets_cropped/images', img))
    height = image.size[0]
    heights.append(height)
    width = image.size[0]
    widths.append(width)
    draw = ImageDraw.Draw(image)
    box_id_i = img[:-3] + 'xml'
    box_id = os.path.join(dir, 'annotations', box_id_i)
    tree = ET.parse(box_id)
    root = tree.getroot()
    for object in root.iter('object'):
        for bndbox in object.iter('bndbox'):
            for x_min in bndbox.iter('xmin'):
                x1 = float(x_min.text)
            for x_max in bndbox.iter('xmax'):
                x2 = float(x_max.text)
            for y_min in bndbox.iter('ymin'):
                y1 = float(y_min.text)
            for y_max in bndbox.iter('ymax'):
                y2 = float(y_max.text)
            boxes.append([x1, y1, x2, y2])
    for elem in range(len(boxes)):
        draw.rectangle([(boxes[elem][0], boxes[elem][1]),
                        (boxes[elem][2], boxes[elem][3])],
                       outline = 'green', width = 3)
    fp = os.path.join('Datasets/GDXray_dataset/datasets_cropped/images_with_boxes', img)
    image.save(fp)


i=0
dir = 'Datasets/GDXray_dataset'
xml_list= os.listdir(os.path.join(dir, 'bboxes_W0001'))
for xml in xml_list:
    tree = ET.parse(os.path.join(dir, 'bboxes_W0001', xml))
    root = tree.getroot()
    for object in root.iter('object'):
        i = i +1