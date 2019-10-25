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
        # this function prepare the dataset to the yolo estructure
        f = open(os.path.join(self.DATASET, self.DATASET_NAME, "classes.names"), "r")
        for line in f:
            annotationParser.CLASSES.append(line.split("\n")[0])
        train_images = list(
            paths.list_files(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train"), validExts=(".jpg")))
        test_images = list(
            paths.list_files(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test"), validExts=(".jpg")))
        traintxt = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train.txt"), "w")
        testtxt = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test.txt"),"w")
        for tr_im in train_images:
            traintxt.write(os.path.abspath(tr_im) + "\n")
        for te_im in test_images:
            testtxt.write(os.path.abspath(te_im) + "\n")
        traintxt.close()
        testtxt.close()
        annotationParser.PascalVOC2YOLO(self.OUTPUT_PATH, self.DATASET_NAME)  # , datasetPath + os.sep + "images"
    def organize(self, train_percentage):
        IObjectDetection.organize(self, train_percentage)
    def createModel(self):
        pass
    def train(self, framework_path = None):
        pass
    def evaluate(self, framework_path = None):
        pass