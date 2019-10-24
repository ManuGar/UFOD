# -*- coding: utf-8 -*-

from datetime import datetime
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from lxml import etree
from imutils import paths

import xml.etree.ElementTree as ElementTree
import shutil
import hashlib
import io
import logging
import os
import contextlib2
import numpy as np
import PIL.Image
import tensorflow as tf

CLASSES = []

# This are the default classes but the classes have to be read from the classes.name file
# [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]


# def PascalVOC2TensorflowRecords(voc_path, images_path, dataset_path, output_path):

def PascalVOC2TensorflowRecords(dataset_path, output_path):
    """Locate files for train and test sets and then generate TFRecords."""
    dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
    result_path = os.path.join(output_path, dataset_name)

    if (not os.path.exists(os.path.join(result_path, "test"))):
        os.makedirs(os.path.join(result_path, "test"))
    if (not os.path.exists(os.path.join(result_path, "train"))):
        os.makedirs(os.path.join(result_path, "train"))

    print('Saving results to {}'.format(result_path))
    dataset = os.path.join(dataset_path, "train")
    anno_paths = [os.path.join(dataset, p) for p in os.listdir(dataset) if p.endswith(".xml")]
    # anno_paths = [os.path.join(voc_path, p) for p in os.listdir(voc_path)]
    image_paths = [os.path.join(dataset, p) for p in os.listdir(dataset) if p.endswith(".jpg")]
    # image_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    # image_train, image_test, anno_train, anno_test = train_test_split(image_paths, anno_paths, test_size=0.25, random_state=42)
    process_dataset(image_paths, anno_paths, result_path,num_shards=1, train=True)

    # for ann in anno_paths:
    #     os.remove(ann)

    dataset2 = os.path.join(dataset_path, "test")
    anno_paths = [os.path.join(dataset2, p) for p in os.listdir(dataset2) if p.endswith(".xml")]
    # anno_paths = [os.path.join(voc_path, p) for p in os.listdir(voc_path)]
    image_paths = [os.path.join(dataset2, p) for p in os.listdir(dataset2) if p.endswith(".jpg")]
    # image_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    # image_train, image_test, anno_train, anno_test = train_test_split(image_paths, anno_paths, test_size=0.25, random_state=42)

    process_dataset(image_paths, anno_paths, result_path, num_shards=1, train=False)

    # for ann in anno_paths:
    #     os.remove(ann)

    # process_dataset('test',image_test, anno_test, test_path,num_shards=1)

    # num_shards es el numero de imagenes que entran dentro de cada archivo de anotaciones de TFRecodrs. Si pones la
    # longitud del dataset, entonces crearas un archivo para cada anotacion
    # process_dataset('test', test_image_paths, test_anno_paths, test_path, num_shards=20)


def PascalVOC2YOLO(dataset_path, output_path, dataset_name):
    listaFicheros_train = list(
        paths.list_files(os.path.join(output_path, dataset_name, "train"), validExts=(".jpg")))
    listaFicheros_test = list(
        paths.list_files(os.path.join(output_path, dataset_name, "test"), validExts=(".jpg")))



    anno_paths = list(paths.list_files(dataset_path, validExts=(".xml")))
    # anno_paths = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path) if p.endswith(".xml")]
    # image_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
    result_path = os.path.join(output_path, dataset_name)      #".." + os.sep + "output" + os.sep + "YOLO"
    if (not os.path.exists(result_path)):
        os.makedirs(os.path.join(result_path, "JPEGImages"))
        os.makedirs(os.path.join(result_path, "Annotations"))


    image_path = os.path.join(dataset_path, "JPEGImages")
    shutil.copy(os.path.join(dataset_path,"classes.names"),result_path)
    for anno in anno_paths:
        name = str(os.path.basename(anno).split('.')[0])
        # name = anno[anno.rfind("/")+1:anno.rfind(".")]
        filename = os.path.join(image_path, name+".jpg")
        # filename = anno[:anno.rfind(".")] + ".jpg"
        shutil.copy(filename,os.path.join(result_path,"JPEGImages"))
        tree = ElementTree.parse(anno)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        dw = 1. / w
        dh = 1. / h
        boxes = process_anno(anno)
        f_name = os.path.join(result_path,"Annotations",str(os.path.basename(anno).split(".")[0])+".txt")
        # f_name = result_path + anno[anno.rfind(os.sep):anno.rfind(".")] + ".txt"
        f = open(f_name, "w")
        for bo in boxes:
            w = bo['x_max'] - bo['x_min']
            h = bo['y_max'] - bo['y_min']
            x = (bo['x_min'] + bo['x_max'])/2.0
            y = (bo['y_min'] + bo['y_max'])/2.0

            x = x * dw
            y = y * dh
            w = w * dw
            h = h * dh
            f.write(str(bo['class']) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
        f.close()
    # Este metodo hay que pasarlo para cada una de las imagenes. Se parsea una por una. Aprovechar y hacerlo para cada
    # imagen que se te tiene que leer. Hacerlo cuando se recorren los archivos vaya


"""Convert Pascal VOC 2007+2012 detection dataset to TFRecords.
Does not preserve full XML annotations.
Combines all VOC 2007 subsets (train, val) with VOC2012 for training.
Uses VOC2012 val for val and VOC2007 test for test.

Code based on:
https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
"""



# Small graph for image decoding
decoder_sess = tf.Session()
image_placeholder = tf.placeholder(dtype=tf.string)
decoded_jpeg = tf.image.decode_jpeg(image_placeholder, channels=3)

def process_dataset(image_paths, anno_paths, result_path, num_shards,train = True):
# def process_dataset(name, image_paths, anno_paths, result_path, num_shards, train=True):

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
        if(train):
            output_filename = 'train.record'
            # output_filename = 'train-{:05d}'.format(shard + 1)
            output_file = os.path.join(os.path.join(result_path), output_filename)
        else:
            output_filename = 'test.record'
            # output_filename = 'test-{:05d}'.format(shard + 1)
            output_file = os.path.join(os.path.join(result_path), output_filename)
        if (not os.path.exists(output_file[:output_file.rfind(os.sep)])):
            os.makedirs(output_file[:output_file.rfind(os.sep)])

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
            example = convert_to_tf(image_data, boxes, image_file, height, width)
            # write to writer
            writer.write(example.SerializeToString())

            if(train):
                shutil.copy(image_paths[i], os.path.join(result_path,"train"))
            else:
                shutil.copy(image_paths[i], os.path.join(result_path,"test"))

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

def dict_to_tf_example(data,
                       mask_path,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False,
                       faces_only=True,
                       mask_type='png'):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  with tf.gfile.GFile(mask_path, 'rb') as fid:
    encoded_mask_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_mask_png)
  mask = PIL.Image.open(encoded_png_io)
  if mask.format != 'PNG':
    raise ValueError('Mask format not PNG')

  mask_np = np.asarray(mask)
  nonbackground_indices_x = np.any(mask_np != 2, axis=0)
  nonbackground_indices_y = np.any(mask_np != 2, axis=1)
  nonzero_x_indices = np.where(nonbackground_indices_x)
  nonzero_y_indices = np.where(nonbackground_indices_y)

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      difficult_obj.append(int(difficult))

      if faces_only:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])
      else:
        xmin = float(np.min(nonzero_x_indices))
        xmax = float(np.max(nonzero_x_indices))
        ymin = float(np.min(nonzero_y_indices))
        ymax = float(np.max(nonzero_y_indices))

      xmins.append(xmin / width)
      ymins.append(ymin / height)
      xmaxs.append(xmax / width)
      ymaxs.append(ymax / height)
      # class_name = get_class_name_from_filename(data['filename'])
      class_name = obj['name']
      classes_text.append(class_name.encode('utf8'))
      classes.append(label_map_dict[class_name])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))
      if not faces_only:
        mask_remapped = (mask_np != 2).astype(np.uint8)
        masks.append(mask_remapped)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }
  if not faces_only:
    if mask_type == 'numerical':
      mask_stack = np.stack(masks).astype(np.float32)
      masks_flattened = np.reshape(mask_stack, [-1])
      feature_dict['image/object/mask'] = (
          dataset_util.float_list_feature(masks_flattened.tolist()))
    elif mask_type == 'png':
      encoded_mask_png_list = []
      for mask in masks:
        img = PIL.Image.fromarray(mask)
        output = io.BytesIO()
        img.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png_list))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,
                     faces_only=True,
                     mask_type='png'):
  """Creates a TFRecord file from examples.
  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
      mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

      if not os.path.exists(xml_path):
        logging.warning('Could not find %s, ignoring example.', xml_path)
        continue
      with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      try:
        tf_example = dict_to_tf_example(
            data,
            mask_path,
            label_map_dict,
            image_dir,
            faces_only=faces_only,
            mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', xml_path)


def main():
    pass
    # PascalVOC2YOLO("../datasets/VOC2012/Annotations")
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")

if __name__ == "__main__":
    main()
