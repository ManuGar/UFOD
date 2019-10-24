import os
import annotationParser
import shutil
from imutils import paths

from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.YOLOObjectDetector import functions as fn


class DarknetAbstract(IObjectDetection):
    def __init__(self,dataset_path, dataset_name):
        IObjectDetection.__init__(self,dataset_path, dataset_name)
    def transform(self):

        # fn.datasetSplit(self.DATASET_NAME, self.OUTPUT_PATH, self.DATASET, train_percentage)

        f = open(os.path.join(self.DATASET, self.DATASET_NAME, "classes.names"), "r")
        if (not os.path.exists(self.OUTPUT_PATH)):
            os.makedirs(self.OUTPUT_PATH)
        for line in f:
            annotationParser.CLASSES.append(line.split("\n")[0])
        annotationParser.PascalVOC2YOLO(self.DATASET, self.OUTPUT_PATH, self.DATASET_NAME)  # , datasetPath + os.sep + "images"
    def organize(self, train_percentage):
        # dataset_name, darknet_path, dataset_path, train%_split
        # this function prepare the dataset to the yolo estructure
        # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        IObjectDetection.organize(self, train_percentage)

        # shutil.copy(os.path.join(self.DATASET[:self.DATASET.rfind(os.sep)], "classes.names"),
        #             os.path.join(self.DATASET, self.DATASET_NAME))
    def createModel(self):
        pass
    def train(self, framework_path = None):
        pass
    def evaluate(self, framework_path = None):
        pass