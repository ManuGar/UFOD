
from mmdetection.mmdet.models import build_detector
from mmdetection.mmdet.apis import inference_detector, show_result, init_detector
from Predictors.IPredictor import IPredictor
from xml.dom import minidom
from imutils import paths
import xml.etree.ElementTree as ET
import numpy as np
import cv2


class mmdetectionPredict(IPredictor):
    CONFIDENCE = 0.5

    def __init__(self, modelWeights, classesFile,model,modelName):
        super().__init__(modelWeights, classesFile)
        self.modelName=modelName
        self.model=model
        LABELS = open(self.classesFile).read().strip().split("\n")
        self.classes = [label.split(',')[0] for label in LABELS]

    def predict(self, imagePaths):
        model = init_detector('mmdetection/configs/'+self.modelName+".py",
                              './work_dirs/'+self.model+'/latest.pth')
        imagePaths = list(paths.list_images(imagePaths))
        for img in imagePaths:
            result = inference_detector(model, img)
            bbox_result = result
            bboxes = np.vstack(bbox_result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32)
                      for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)
            scores = bboxes[:, -1]
            inds = scores > self.CONFIDENCE
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            image = cv2.imread(img)
            (hI, wI, d) = image.shape

            xmlPath = img[0:img.rfind(".")] + ".xml"
            file = open(xmlPath, "w")
            file.write(self.generateXML(img.split("/")[-1], img[0:img.rfind("/")], wI, hI, d, bboxes,scores,labels,self.classes))
            file.close()






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
            (x, y, xmax, ymax,_) = box
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



