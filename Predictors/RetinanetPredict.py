# USAGE
# python predict_batch.py --model output.h5 --labels logos/retinanet_classes.csv
#	--input logos/images --output output

# import the necessary packages
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
from Predictors.IPredictor import IPredictor
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths
import numpy as np
import argparse
import os


class RetinanetPredictor(IPredictor):
    CONFIDENCE = 0.5

    def __init__(self, modelWeights, classesFile):
        super().__init__(modelWeights, classesFile)
        f = open(self.classesFile, 'rt')
        self.labels = {i: L for i,L in enumerate(self.LABELS)}
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            self.LABELS = open(classesFile).read().strip().split("\n")
            self.LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in self.LABELS}

    def predict(self, imagePaths):
        # TODO:
        # Allow option for --input to be a .txt file OR a directory. Check if
        # file, and if so, presume keras-retinanet set of images + labels

        # load the class label mappings

        # load the model from disk and grab all input image paths
        model = models.load_model(self.modelWeights, backbone_name="resnet50")
        imagePaths = list(paths.list_images(imagePaths))

        for (i, imagePath) in enumerate(imagePaths):
            # load the input image (in BGR order), clone it, and preprocess it
            print("[INFO] predicting on image {} of {}".format(i + 1,
                                                               len(imagePaths)))
            # load the input image (in BGR order), clone it, and preprocess it
            image = read_image_bgr(imagePath)
            wI, hI, d = image.shape
            output = image.copy()
            image = preprocess_image(image)
            (image, scale) = resize_image(image)
            image = np.expand_dims(image, axis=0)

            # detect objects in the input image and correct for the image scale
            (boxes, scores, labels) = model.predict_on_batch(image)
            boxes /= scale
            boxes1 = []
            for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
                if score < self.CONFIDENCE:
                    continue
                boxes1.append(([label, box], score))

            # parse the filename from the input image path, construct the
            # path to the output image, and write the image to disk
            filename = imagePath.split(os.path.sep)[-1]
            # outputPath = os.path.sep.join([args["output"], filename])

            file = open(imagePath[0:imagePath.rfind(".")] + ".xml", "w")
            file.write(self.generateXML(imagePath[0:imagePath.rfind(".")], imagePath, hI, wI, d, boxes1))
            file.close()
        # cv2.imwrite(outputPath, output)

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
        for (box, score) in boxes:
            category = self.LABELS[box[0]]
            box = box[1].astype("int")
            (x, y, xmax, ymax) = box
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


# loop over the input image paths
