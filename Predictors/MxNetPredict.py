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

class MxNetPredict(IPredictor):
    CONFIDENCE=0.5
    def __init__(self,modelWeights,classesFile, model):
        super().__init__(modelWeights,classesFile)
        self.model = model
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def predict(self, imagePaths):
        net = gcv.model_zoo.get_model(self.model, classes=self.classes, pretrained_base=False)
        net.load_parameters(self.modelWeights)
        imagePaths = list(paths.list_images(imagePaths))

        print(imagePaths)
        for (i, imagePath) in enumerate(imagePaths):
            # load the input image (in BGR order), clone it, and preprocess it
            # print("[INFO] predicting on image {} of {}".format(i + 1,
            #	len(imagePaths)))

            # load the input image (in BGR order), clone it, and preprocess it
            image = cv2.imread(imagePath)
            (hI, wI, d) = image.shape

            # detect objects in the input image and correct for the image scale
            # Poner short=512
            x, image = gcv.data.transforms.presets.ssd.load_test(imagePath, min(wI, hI), max_size=max(wI, hI))
            cid, score, bbox = net(x)
            boxes1 = []
            # Añadir cid[0]
            for (cid, box, score) in zip(cid[0], bbox[0], score[0]):
                if score < self.CONFIDENCE:
                    continue
                # Añadir label que sera con net.classes[cid]
                boxes1.append(([net.classes[cid[0].asnumpy()[0].astype('int')], box], score))

            # parse the filename from the input image path, construct the
            # path to the output image, and write the image to disk
            filename = imagePath.split(os.path.sep)[-1]
            # outputPath = os.path.sep.join([args["output"], filename])
            file = open(imagePath[0:imagePath.rfind(".")] + ".xml", "w")
            file.write(self.generateXML(imagePath[0:imagePath.rfind(".")], imagePath, wI, hI, d, boxes1))
            file.close()

        # cv2.imwrite(outputPath, output)

    def prettify(self,elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def generateXML(self,filename, outputPath, w, h, d, boxes):
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
            (x, y, xmax, ymax) = box
            childObject = ET.SubElement(top, 'object')
            childName = ET.SubElement(childObject, 'name')
            childName.text = category
            childScore = ET.SubElement(childObject, 'confidence')
            childScore.text = str(score.asscalar())
            childPose = ET.SubElement(childObject, 'pose')
            childPose.text = 'Unspecified'
            childTruncated = ET.SubElement(childObject, 'truncated')
            childTruncated.text = '0'
            childDifficult = ET.SubElement(childObject, 'difficult')
            childDifficult.text = '0'
            childBndBox = ET.SubElement(childObject, 'bndbox')
            childXmin = ET.SubElement(childBndBox, 'xmin')
            childXmin.text = str(x.asscalar())
            childYmin = ET.SubElement(childBndBox, 'ymin')
            childYmin.text = str(y.asscalar())
            childXmax = ET.SubElement(childBndBox, 'xmax')
            childXmax.text = str(xmax.asscalar())
            childYmax = ET.SubElement(childBndBox, 'ymax')
            childYmax.text = str(ymax.asscalar())
        return self.prettify(top)