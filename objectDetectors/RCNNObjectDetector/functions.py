
import xml.etree.ElementTree as ElementTree
import os
from imutils import paths
import shutil

def obtain_box(anno_path):
    """Process Pascal VOC annotations."""
    with open(anno_path) as f:
        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    size = root.find('size')
    height = float(size.find('height').text)
    width = float(size.find('width').text)
    boxes = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        xml_box = obj.find('bndbox')
        bbox = {
            'class': label,
            'y_min': float(xml_box.find('ymin').text),
            'x_min': float(xml_box.find('xmin').text),
            'y_max': float(xml_box.find('ymax').text),
            'x_max': float(xml_box.find('xmax').text)
        }
        boxes.append(bbox)

    return boxes

def extract_boxes(anno_path):
    """Process Pascal VOC annotations."""
    with open(anno_path) as f:

        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    boxes = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        xml_box = obj.find('bndbox')
        bbox = {
            'class': label,
            'y_min': round(float(xml_box.find('ymin').text)),
            'x_min': round(float(xml_box.find('xmin').text)),
            'y_max': round(float(xml_box.find('ymax').text)),
            'x_max': round(float(xml_box.find('xmax').text))
        }

        boxes.append(bbox)
    return boxes, width, height

def count_classes(classes):
    return len(classes)

def organizeDataset(Nproyecto, output_path, dataset_path):
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    listaFicheros = list(paths.list_files(os.path.join(dataset_path,"dataset"), validExts=(".jpg")))
    shutil.copy(os.path.join(dataset_path,"classes.names"), output_path)

    # creamos la estructura de carpetas, la primera contendra las imagenes del entrenamiento
    if (not os.path.exists(os.path.join(output_path, Nproyecto, 'images'))):
        os.makedirs(os.path.join(output_path, Nproyecto, 'images'))
    if (not os.path.exists(os.path.join(output_path, Nproyecto, 'annots'))):
        os.makedirs(os.path.join(output_path, Nproyecto, 'annots'))
    # esta carpeta contendra las anotaciones de las imagenes de entrenamiento
    # os.makedirs(os.path.join(output_path, Nproyecto, 'annotations'))
    # y esta ultima carpeta va a contener tanto las imagenes como los ficheros de anotaciones del test
    # para las imagenes de entrenamiento
    for file in listaFicheros:
        name = os.path.basename(file).split('.')[0]
        annotation_path = file[:file.rfind(os.sep) + 1] + name + ".xml"
        new_image_path = os.path.join(output_path, Nproyecto, 'images', name + '.jpg')
        new_annotation_path = os.path.join(output_path, Nproyecto, 'annots', name + '.xml')
        shutil.copy(file, new_image_path)
        shutil.copy(annotation_path, new_annotation_path)

