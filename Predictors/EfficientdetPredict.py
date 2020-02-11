# USAGE
# python predict_batch.py --input logos/images --output output
from Predictors.IPredictor import IPredictor
from xml.dom import minidom
from imutils import paths
from mxnet import autograd, gluon
from mxnet import autograd, gluon
import numpy as np
import mxnet as mx
import gluoncv as gcv
import xml.etree.ElementTree as ET
import cv2
import os
from EfficientDet.model import efficientdet
from EfficientDet.utils import preprocess_image
from EfficientDet.utils.anchors import anchors_for_shape
import time


class EfficientdetPredict(IPredictor):
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    CONFIDENCE=0.5
    def __init__(self,modelWeights,classesFile, model):
        super().__init__(modelWeights,classesFile)
        self.model = model
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def predict(self, imagePaths):

        image_size = self.image_sizes[self.model]
        LABELS = open(self.classesFile).read().strip().split("\n")
        classes = [label.split(',')[0] for label in LABELS]
        num_classes = len(classes)
        weighted_bifpn = False
        colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
        model, prediction_model = efficientdet(phi=self.model,
                                               weighted_bifpn=weighted_bifpn,
                                               num_classes=num_classes,
                                               score_threshold=self.CONFIDENCE)
        prediction_model.load_weights(self.modelWeights, by_name=True)

        imagePaths = list(paths.list_images(imagePaths))

        for (i, image_path) in enumerate(imagePaths):
            print("[INFO] predicting on image {} of {}".format(i + 1, len(imagePaths)))
            image = cv2.imread(image_path)
            src_image = image.copy()
            image = image[:, :, ::-1]
            h, w = image.shape[:2]

            image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
            inputs = np.expand_dims(image, axis=0)
            anchors = anchors_for_shape((image_size, image_size))
            # run network
            start = time.time()
            boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                                       np.expand_dims(anchors, axis=0)])
            boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] - offset_w
            boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] - offset_h
            boxes /= scale
            boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
            boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
            boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
            boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > self.CONFIDENCE)[0]

            # select those detections
            boxes = boxes[0, indices]
            scores = scores[0, indices]
            labels = labels[0, indices]

            file = open(image_path[0:image_path.rfind(".")] + ".xml", "w")
            file.write(
                self.generateXML(image_path[0:image_path.rfind(".")], image_path, h, w, 3, boxes, scores, labels, classes))
            file.close()

    def predictImage(self, imagePath):
        net = gcv.model_zoo.get_model(self.model, classes=self.classes, pretrained_base=False)
        net.load_parameters(self.modelWeights)
        # load the input image (in BGR order), clone it, and preprocess it
        image = cv2.imread(imagePath)
        xmlPath = imagePath[0:imagePath.rfind(".")] + ".xml"
        (hI, wI, d) = image.shape

        # detect objects in the input image and correct for the image scale
        # Poner short=512
        x, image = gcv.data.transforms.presets.ssd.load_test(imagePath, 512)
        cid, score, bbox = net(x)
        (HI, WI, d) = image.shape
        boxes1 = []
        # Añadir cid[0]
        for (cid, box, score) in zip(cid[0], bbox[0], score[0]):
            if score < self.CONFIDENCE:
                continue
            # Añadir label que sera con net.classes[cid]
            (x, y, xmax, ymax) = box.asnumpy()
            box = (x * wI / WI, y * hI / HI, xmax * wI / WI, ymax * hI / HI)
            boxes1.append(([net.classes[cid[0].asnumpy()[0].astype('int')], box], score))
        # parse the filename from the input image path, construct the
        # path to the output image, and write the image to disk
        filename = imagePath.split(os.path.sep)[-1]
        # outputPath = os.path.sep.join([args["output"], filename])
        file = open(xmlPath, "w")
        file.write(self.generateXML(imagePath.split("/")[-1], imagePath[0:imagePath.rfind("/")], wI, hI, d, boxes1))
        file.close()
        self.combineImageAndPrediction(imagePath, xmlPath)



    def generateXML(self,filename, outputPath, w, h, d, boxes, scores, labels, classes):
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
        for (category, box, score) in zip(labels, boxes, scores):
            (x, y, xmax, ymax) = box
            childObject = ET.SubElement(top, 'object')
            childName = ET.SubElement(childObject, 'name')
            childName.text = classes[category]
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
            childXmin.text = str(int(x))
            childYmin = ET.SubElement(childBndBox, 'ymin')
            childYmin.text = str(int(y))
            childXmax = ET.SubElement(childBndBox, 'xmax')
            childXmax.text = str(int(xmax))
            childYmax = ET.SubElement(childBndBox, 'ymax')
            childYmax.text = str(int(ymax))
        return self.prettify(top)

    def prettify(self,elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")