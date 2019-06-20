import os
import annotationParser

from objectDetectors.objectDetectionInterface import IObjectDetection

class DarknetAbstract(IObjectDetection):
    def __init__(self):
        IObjectDetection.__init__(self)
        pass
    def transform(self, dataset_path, classes_path, output_path):
        f = open(classes_path, "r")
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        classes_name = open(os.path.join(output_path, "classes.names"), "w")
        for line in f:
            annotationParser.CLASSES.append(line.split("\n")[0])
            classes_name.write(line)
        annotationParser.PascalVOC2YOLO(dataset_path, output_path)  # , datasetPath + os.sep + "images"
    def organize(self, datasetPath, output_path, train_percentage):
        pass
    def createModel(self, datasetPath):
        pass
    def train(self, dataset_path):
        pass
    def evaluate(self, dataset_path):
        pass