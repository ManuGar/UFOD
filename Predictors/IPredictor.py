import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom


class IPredictor(object):
    def __init__(self,modelWeights,classesFile):
        self.modelWeights = modelWeights
        self.classesFile = classesFile
    def predict(self, imagePaths):
        pass
    def predictImage(self, imagePath):
        pass



    def combineImageAndPrediction(imagePath, xmlPath):
        image = cv2.imread(imagePath)
        tree = ET.parse(xmlPath)
        root = tree.getroot()
        objects = root.findall('object')
        boxes = []
        for object in objects:
            category = object.find('name').text
            confidence = object.find('confidence')
            box = object.find('bndbox')
            x = int(box.find('xmin').text)
            y = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)
            xmax = int(box.find('xmax').text)
            cv2.rectangle(image, (x, y), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, category, (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite('prediction.jpg', image)

    def prettify(self,elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
