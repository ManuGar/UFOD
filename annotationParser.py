# -*- coding: utf-8 -*-


from imutils import paths
import os
import xml.etree.ElementTree as ElementTree



# This are the default classes but the classes have to be read from the classes.name file
# [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]


# def PascalVOC2TensorflowRecords(voc_path, images_path, dataset_path, output_path):
CLASSES= []
def process_anno(anno_path):
    """Process Pascal VOC annotations."""
    with open(anno_path) as f:
        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    size = root.find('size')
    height = float(size.find('height').text)
    width = float(size.find('width').text)
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in CLASSES or int(difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = {
            'class': CLASSES.index(label),
            'y_min': float(xml_box.find('ymin').text) / height,
            'x_min': float(xml_box.find('xmin').text) / width,
            'y_max': float(xml_box.find('ymax').text) / height,
            'x_max': float(xml_box.find('xmax').text) / width
        }
        boxes.append(bbox)
    return boxes

def PascalVOC2YOLO(dataset_path, dataset_name):
    listaFicheros_train = list(
        paths.list_files(os.path.join(dataset_path, dataset_name, "train"), validExts=(".xml")))
    listaFicheros_test = list(
        paths.list_files(os.path.join(dataset_path, dataset_name, "test"), validExts=(".xml")))
    result_path = os.path.join(dataset_path, dataset_name)
    if (not (os.path.exists(os.path.join(result_path,"train","labels"))) and not(os.path.exists(os.path.join(result_path,"train","labels")))):
        os.makedirs(os.path.join(result_path,"train","labels"))
        os.makedirs(os.path.join(result_path,"test","labels"))
    for anno in listaFicheros_train:
        write_anno(anno,os.path.join(result_path,"train","labels"))
    for anno_test in listaFicheros_test:
        write_anno(anno_test, os.path.join(result_path,"test","labels"))

def write_anno(anno_path, result_path):
    tree = ElementTree.parse(anno_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    dw = 1. / w
    dh = 1. / h
    boxes = process_anno(anno_path)
    f_name = os.path.join(result_path, str(os.path.basename(anno_path).split(".")[0]) + ".txt")
    f = open(f_name, "w")
    for bo in boxes:
        w = bo['x_max'] - bo['x_min']
        h = bo['y_max'] - bo['y_min']
        x = (bo['x_min'] + bo['x_max']) / 2.0
        y = (bo['y_min'] + bo['y_max']) / 2.0

        x = x * dw
        y = y * dh
        w = w * dw
        h = h * dh
        f.write(str(bo['class']) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
    f.close()

"""Convert Pascal VOC 2007+2012 detection dataset to TFRecords.
Does not preserve full XML annotations.
Combines all VOC 2007 subsets (train, val) with VOC2012 for training.
Uses VOC2012 val for val and VOC2007 test for test.

Code based on:
https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
"""





def main():
    pass
    # PascalVOC2YOLO("../datasets/VOC2012/Annotations")
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")

if __name__ == "__main__":
    main()
