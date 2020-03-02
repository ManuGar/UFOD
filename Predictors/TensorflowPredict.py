# USAGE
from Predictors.IPredictor import IPredictor
from xml.dom import minidom
from imutils import paths
import tensorflow as tf
import numpy as np
import glob
import xml.etree.ElementTree as ET
import cv2
import os
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from PIL import Image





class TensorflowPredict(IPredictor):
    CONFIDENCE=0.5
    def __init__(self,modelWeights,classesFile, model):
        super().__init__(modelWeights,classesFile)
        self.model = model
        # with open(self.classesFile, 'rt') as f:
        #     self.classes = f.read().rstrip('\n').split('\n')

    def predict(self, imagePaths):
        aux_path = self.classesFile[:self.classesFile.rfind(os.sep)]

        test_record_fname = os.join(aux_path,"test.record")
        train_record_fname =  os.join(aux_path,"train.record")
        label_map_pbtxt_fname = self.classesFile
        output_directory = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "models")
        pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph.pb")
        assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = pb_fname

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = label_map_pbtxt_fname

        # If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
        PATH_TO_TEST_IMAGES_DIR = os.path.join('/content/Optic', "test")

        assert os.path.isfile(pb_fname)
        assert os.path.isfile(PATH_TO_LABELS)
        TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.jpg"))
        assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
        print(TEST_IMAGE_PATHS)


        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        num_classes = self.get_num_classes(label_map_pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        for i, image_path in enumerate(imagePaths):
            print(str(i) + "/" + str(len(imagePaths)))
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np, detection_graph)
            (h, w, d) = (image_np.shape)
            # Visualization of the results of a detection.
            boxes = []
            for b, c, s in zip(output_dict['detection_boxes'], output_dict['detection_classes'],
                               output_dict['detection_scores']):
                if s > 0.5:
                    ymin, xmin, ymax, xmax = b
                    ymin = ymin * h
                    xmin *= w
                    xmax *= w
                    ymax *= h
                    cat = (category_index[c])['name']
                    boxes.append([cat, np.array([ymin, xmin, ymax, xmax])])
            imagePath = image_path
            filename = imagePath.split(os.path.sep)[-1]
            file = open(imagePath[0:imagePath.rfind(".")] + ".xml", "w")
            file.write(self.generateXML(imagePath[0:imagePath.rfind(".")], imagePath, w, h, d,
                                   zip(boxes, output_dict['detection_scores'])))
            file.close()


    def prettify(self,elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def generateXML(self, filename, outputPath, w, h, d, boxes):
        top = ET.Element('annotation')
        childFolder = ET.SubElement(top, 'folder')
        childFolder.text = 'images'
        childFilename = ET.SubElement(top, 'filename')
        childFilename.text = filename[0:filename.rfind(".")]
        childPath = ET.SubElement(top, 'path')
        childPath.text = outputPath + "/" + filename
        childSource = ET.SubElement(top, 'source')
        childDatabase = ET.SubElement(childSource, 'database')
        childDatabase.text = 'Unknown'
        childSize = ET.SubElement(top, 'size')
        childWidth = ET.SubElement(childSize, 'width')
        childWidth.text = str(w)
        childHeight = ET.SubElement(childSize, 'height')
        childHeight.text = str(h)
        childDepth = ET.SubElement(childSize, 'depth')
        childDepth.text = str(d)
        childSegmented = ET.SubElement(top, 'segmented')
        childSegmented.text = str(0)
        # boxes tiene que contener labels
        for (box, score) in boxes:
            # Cambiar categoria por label
            category = box[0]
            box = box[1].astype("int")
            #######
            # Cuidado esto está cambiado con respecto a lo que es habitualmente
            #######
            (y, x, ymax, xmax) = box
            childObject = ET.SubElement(top, 'object')
            childName = ET.SubElement(childObject, 'name')
            childName.text = category
            childScore = ET.SubElement(childObject, 'confidence')
            childScore.text = str(score)
            childPose = ET.SubElement(childObject, 'pose')
            childPose.text = 'Unspecified'
            childTruncated = ET.SubElement(childObject, 'truncated')
            childTruncated.text = '0'
            childDifficult = ET.SubElement(childObject, 'difficult')
            childDifficult.text = '0'
            childBndBox = ET.SubElement(childObject, 'bndbox')
            childXmin = ET.SubElement(childBndBox, 'xmin')
            childXmin.text = str(x)
            childYmin = ET.SubElement(childBndBox, 'ymin')
            childYmin.text = str(y)
            childXmax = ET.SubElement(childBndBox, 'xmax')
            childXmax.text = str(xmax)
            childYmax = ET.SubElement(childBndBox, 'ymax')
            childYmax.text = str(ymax)
        return self.prettify(top)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                        real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                        real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def get_num_classes(self, pbtxt_fname):
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
