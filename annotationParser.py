import argparse
import os
import xml.etree.ElementTree as ElementTree
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from datetime import datetime

def PascalVOC2TensorflowRecords(voc_path, images_path):
    """Locate files for train and test sets and then generate TFRecords."""

    voc_path = os.path.expanduser(voc_path)
    result_path = os.path.join("../output2", 'TFRecords')
    print('Saving results to {}'.format(result_path))

    train_path = os.path.join(result_path, 'train')
    test_path = os.path.join(result_path, 'test')

    anno_paths = [os.path.join(voc_path, p) for p in os.listdir(voc_path)]
    image_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]

    # image_train, image_test, anno_train, anno_test = train_test_split(image_paths, anno_paths, test_size=0.25, random_state=42)

    process_dataset('train',image_paths, anno_paths, train_path,num_shards=1)
    # process_dataset('test',image_test, anno_test, test_path,num_shards=1)

    # num_shards es el número de imágenes que entran dentro de cada archivo de anotaciones de TFRecodrs. Si pones la
    # longitud del dataset, entonces crearas un archivo para cada anotacion
    # process_dataset('test', test_image_paths, test_anno_paths, test_path, num_shards=20)


# Falta de modificar al ruta de salida para que sea justo la del nombre del dataset o la que me diga jonathan de poner
# ademas de saber si tambien tenemos que copiar el dataset de imagenes a esa nueva ruta con las anotaciones
def PascalVOC2YOLO(voc_path, images_path):
    anno_paths = [os.path.join(voc_path, p) for p in os.listdir(voc_path)]
    image_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    result_path = "../output2/train"

    for anno in anno_paths:
        tree = ElementTree.parse(anno)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        boxes = process_anno(anno)
        f = open(result_path + anno[anno.rfind(os.sep):] + ".txt")
        for bo in boxes:
            f.write(bo[0] + " " + str((bo[2] + bo[4])/2.0) + " " + str((bo[1] + bo[3])/2.0)  + str(w) + " " + str(h) )
        f.close()


    # Este metodo hay que pasarlo para cada una de las imagenes. Se parseta una por una. Aprovechar y hacerlo para cada
    # imagen que se te tiene que leer. Hacerlo cuando se recorren los archivos vaya

def PascalVOC2OpenCV():
    pass
def PascalVOC2Protobuf():
    pass
def YOLO2Protobuf():
    pass
def Protobuf2PascalVOC():
    pass
def OpenCV2PascalVOC():
    pass
def TensorflowRecords2PascalVOC():
    pass



# Luego si es necesario aniadir los pasos que quedan que simplemente seria usar los cambios que tenemos, pasando siempre por PascalVOC








# <annotation>
# 	<folder>GeneratedData_Train</folder>
# 	<filename>000001.png</filename>
# 	<path>/my/path/GeneratedData_Train/000001.png</path>
# 	<source>
# 		<database>Unknown</database>
# 	</source>
# 	<size>
# 		<width>224</width>
# 		<height>224</height>
# 		<depth>3</depth>
# 	</size>
# 	<segmented>0</segmented>
# 	<object>
# 		<name>21</name>
# 		<pose>Frontal</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<occluded>0</occluded>
# 		<bndbox>
# 			<xmin>82</xmin>
# 			<xmax>172</xmax>
# 			<ymin>88</ymin>
# 			<ymax>146</ymax>
# 		</bndbox>
# 	</object>
# </annotation>
















"""Convert Pascal VOC 2007+2012 detection dataset to TFRecords.
Does not preserve full XML annotations.
Combines all VOC 2007 subsets (train, val) with VOC2012 for training.
Uses VOC2012 val for val and VOC2007 test for test.

Code based on:
https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
"""

# Estas clases se deberan coger de un archivo que se llamara siempre igual para que luego aqui se cojan y se compruebe para la dificultad
# el archivo tendra siempre el formato .names para poder leer las clases correctamente
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Small graph for image decoding
decoder_sess = tf.Session()
image_placeholder = tf.placeholder(dtype=tf.string)
decoded_jpeg = tf.image.decode_jpeg(image_placeholder, channels=3)

def process_dataset(name, image_paths, anno_paths, result_path, num_shards):
    """Process selected Pascal VOC dataset to generate TFRecords files.

    Parameters
    ----------
    name : string
        Name of resulting dataset 'train' or 'test'.
    image_paths : list
        List of paths to images to include in dataset.
    anno_paths : list
        List of paths to corresponding image annotations.
    result_path : string
        Path to put resulting TFRecord files.
    num_shards : int
        Number of shards to split TFRecord files into.
    """
    shard_ranges = np.linspace(0, len(image_paths), num_shards + 1).astype(int)
    counter = 0

    for shard in range(num_shards):
        # Generate shard file name
        output_filename = '{}-{:05d}-of-{:05d}'.format(name, shard, num_shards)
        output_file = os.path.join(result_path, output_filename)
        if (not os.path.exists(output_file[:output_file.rfind("/")])):
            os.makedirs(output_file[:output_file.rfind("/")])

        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = range(shard_ranges[shard], shard_ranges[shard + 1])
        for i in files_in_shard:
            image_file = image_paths[i]
            anno_file = anno_paths[i]

            # processes image + anno
            image_data, height, width = process_image(image_file)
            boxes = process_anno(anno_file)

            # convert to example
            example = convert_to_tf(image_data, boxes, image_file, height,
                                         width)

            # write to writer
            writer.write(example.SerializeToString())

            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('{} : Processed {:d} of {:d} images.'.format(
                    datetime.now(), counter, len(image_paths)))
        writer.close()
        print('{} : Wrote {} images to {}'.format(
            datetime.now(), shard_counter, output_filename))

    print('{} : Wrote {} images to {} shards'.format(datetime.now(), counter,
                                                     num_shards))

def convert_to_tf(image_data, boxes, filename, height, width):
    """Convert Pascal VOC ground truth to TFExample protobuf.

    Parameters
    ----------
    image_data : bytes
        Encoded image bytes.
    boxes : dict
        Bounding box corners and class labels
    filename : string
        Path to image file.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    example : protobuf
        Tensorflow Example protobuf containing image and bounding boxes.
    """
    box_classes = [b['class'] for b in boxes]
    box_ymin = [b['y_min'] for b in boxes]
    box_xmin = [b['x_min'] for b in boxes]
    box_ymax = [b['y_max'] for b in boxes]
    box_xmax = [b['x_max'] for b in boxes]
    encoded_image = [tf.compat.as_bytes(image_data)]
    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes':
        tf.train.Feature(int64_list=tf.train.Int64List(value=box_classes)),
        'y_mins':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_ymin)),
        'x_mins':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_xmin)),
        'y_maxes':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_ymax)),
        'x_maxes':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_xmax)),
        'encoded':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))
    return example






















# Auxiliary functions to process the images and annotations of pascalvoc. I think it can be used to process the dataset
# and parse to the rest of annotations

def process_image(image_path):
    """Decode image at given path."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image = decoder_sess.run(decoded_jpeg,
                             feed_dict={image_placeholder: image_data})
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[2]
    assert image.shape[2] == 3
    return image_data, height, width


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
        if label not in classes or int(difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = {
            'class': classes.index(label),
            'y_min': float(xml_box.find('ymin').text) / height,
            'x_min': float(xml_box.find('xmin').text) / width,
            'y_max': float(xml_box.find('ymax').text) / height,
            'x_max': float(xml_box.find('xmax').text) / width
        }
        boxes.append(bbox)
    return boxes









