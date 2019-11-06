from Predictors.IPredictor import IPredictor
from xml.dom import minidom
import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np
from imutils import paths
import argparse
import os

class DarknetPredict(IPredictor):
    CONFTRESHOLD = 0.25  # Confidence threshold
    NMSTRESHOLD = 0.45  # Non-maximum suppression threshold
    INPWIDTH = 416  # Width of network's input image
    INPHEIGHT = 416  # Height of network's input image
    def __init__(self, modelWeights,classesFile,modelConfiguration):
        super().__init__(modelWeights,classesFile)
        self.modelConfiguration = modelConfiguration
        # classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # Give the configuration and weight files for the model and
        # load the network using them.
        self.net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    def predict(self, imagePaths):
        for i, image in enumerate(imagePaths):
            try:
                generateXMLFromImage(image)
            except:
                print('Error ' + image)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Remove the bounding boxes with low confidence using non-maxima suppression
# def postprocess(imagePath,frame, outs):

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > DarknetPredict.CONFTRESHOLD:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # indices = cv.dnn.NMSBoxes(boxes, confidences, DarknetPredict.CONFTRESHOLD, DarknetPredict.NMSTRESHOLD)
    # for i in indices:
    #     i = i[0]
    #     box = boxes[i]
    #     left = box[0]
    #     top = box[1]
    #     width = box[2]
    #     height = box[3]
        #drawPred(frame,classIds[i], confidences[i], left, top, left + width, top + height)
        #cv.imwrite(imagePath[0:imagePath.rfind(".")]+"_res.jpg",frame)
    # Cambiado para que tome valores de confianza.
    return boxes,confidences,classIds

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# Cambiado para que tome valores de confianza.
def generateXML(filename, outputPath, w, h, d, boxes, confidences, classIds):
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
    # Cambiado para que tome valores de confianza.
    for box, confidence, categoryID in zip(boxes, confidences, classIds):
        category = DarknetPredict.classes[categoryID]
        (x, y, wb, hb) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = category
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        # Cambiado para que tome valores de confianza.
        childConfidence = ET.SubElement(childObject, 'confidence')
        childConfidence.text = str(confidence)
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(x)
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(y)
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(x + wb)
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(y + hb)
    return prettify(top)

# In[8]:

def generateXMLFromImage(imagePath):
    im = cv.VideoCapture(imagePath)
    hasFrame, frame = im.read()
    # adding the border
    # (h,w)=frame.shape[:2]
    # WHITE=[255,255,255]
    # if h > w:
    #    frame= cv.copyMakeBorder(frame,0,0,0,h-w,cv.BORDER_CONSTANT,value=WHITE)
    # else:
    #    frame= cv.copyMakeBorder(frame,0,w-h,0,0,cv.BORDER_CONSTANT,value=WHITE)

    # frame = cv.flip( frame, 0 )
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (DarknetPredict.INPWIDTH, DarknetPredict.INPHEIGHT), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    DarknetPredict.net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = DarknetPredict.net.forward(getOutputsNames(DarknetPredict.net))

    # Remove the bounding boxes with low confidence
    # Cambiado para que tome valores de confianza.
    boxes, confidences, classIds = postprocess(frame, outs)
    # print(len(boxes))

    wI, hI, d = frame.shape
    file = open(imagePath[0:imagePath.rfind(".")] + ".xml", "w")
    # file = open(imagePath[0:imagePath.rfind(".")]+".xml", "w")
    file.write(generateXML(imagePath[0:imagePath.rfind(".")], "", wI, hI, d, boxes, confidences, classIds))
    file.close()
    # cv.imwrite(imagePath,frame)




