import os
import annotationParser
import shutil

from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.YOLOObjectDetector import functions as fn


class DarknetAbstract(IObjectDetection):
    def __init__(self):
        IObjectDetection.__init__(self)
    def transform(self, dataset_path, output_path):
        f = open(os.path.join(dataset_path, "classes.names"), "r")
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        classes_name = open(os.path.join(output_path, "classes.names"), "w")
        for line in f:
            annotationParser.CLASSES.append(line.split("\n")[0])
            classes_name.write(line)
        annotationParser.PascalVOC2YOLO(os.join(dataset_path, "dataset"), output_path)  # , datasetPath + os.sep + "images"
    def organize(self, dataset_path, output_path, train_percentage):
        # dataset_name, darknet_path, dataset_path, train%_split
        # this function prepare the dataset to the yolo estructure
        dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        fn.datasetSplit(dataset_name, output_path, dataset_path, train_percentage)
        shutil.copy(os.path.join(dataset_path[:dataset_path.rfind(os.sep)], "classes.names"),
                    os.path.join(output_path, dataset_name))
    def createModel(self, datasetPath):
        pass
    def train(self, dataset_path):
        pass
    def evaluate(self, dataset_path):
        pass