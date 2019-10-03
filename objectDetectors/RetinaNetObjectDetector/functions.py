from sklearn.model_selection import train_test_split
from imutils import paths

import os
import shutil
import xml.etree.ElementTree as ElementTree



def datasetSplit(Nproyecto, output_path, pathImages, porcentaje):
    listaFicheros = list(paths.list_files(pathImages, validExts=(".jpg")))
    train_list, test_list, _, _ = train_test_split(listaFicheros, listaFicheros, train_size=porcentaje,random_state=5)
    # creamos la estructura de carpetas, la primera contendra las imagenes del entrenamiento
    if (not os.path.exists(os.path.join(output_path, Nproyecto, 'images'))):
        os.makedirs(os.path.join(output_path, Nproyecto, 'images'))
    # esta carpeta contendra las anotaciones de las imagenes de entrenamiento
    # os.makedirs(os.path.join(output_path, Nproyecto, 'annotations'))
    # y esta ultima carpeta va a contener tanto las imagenes como los ficheros de anotaciones del test
    # para las imagenes de entrenamiento
    traincsv = open(os.path.join(output_path, Nproyecto, Nproyecto + "_train.csv"),"w")
    testcsv = open(os.path.join(output_path, Nproyecto, Nproyecto + "_test.csv"),"w")
    classes = set()
    for file in train_list:
        name = os.path.basename(file).split('.')[0]
        annotation_file = file[:file.rfind(os.sep) + 1] + name + ".xml"
        image_splited = os.path.join(output_path, Nproyecto, 'images', name + '.jpg')
        boxes = obtain_box(annotation_file)
        for bo in boxes:
            traincsv.write(os.path.abspath(image_splited) + "," + str(int(bo["x_min"])) + "," + str(int(bo["y_min"])) + ","
                           + str(int(bo["x_max"])) + "," + str(int(bo["y_max"])) + "," + str(bo["class"]) + "\n")
            classes.add(bo["class"])
        shutil.copy(file, image_splited)
    # para las imagenes de entrenamiento
    for file in test_list:
        name = os.path.basename(file).split('.')[0]
        annotation_file = file[:file.rfind(os.sep) + 1] + name + ".xml"
        image_splited = os.path.join(output_path, Nproyecto, 'images', name + '.jpg')
        boxes = obtain_box(annotation_file)
        for bo in boxes:
            testcsv.write(os.path.abspath(image_splited) + "," + str(int(bo["x_min"])) + "," + str(int(bo["y_min"])) + ","
                           + str(int(bo["x_max"])) + "," + str(int(bo["y_max"])) + "," + str(bo["class"]) + "\n")
            classes.add(bo["class"])
        shutil.copy(file, image_splited)

    classescsv = open(os.path.join(output_path, Nproyecto, Nproyecto + "_classes.csv"),"w")
    rows = [",".join([c, str(i)]) for (i, c) in enumerate(classes)]
    classescsv.write("\n".join(rows))
    classescsv.close()
    traincsv.close()
    testcsv.close()


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
