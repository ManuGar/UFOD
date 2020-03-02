import os
import objectDetectors.TensorflowObjectDetector.functions as fn
class SSDobjectDetector():


    def transform(datasetPath):
        fn.PascalVOC2TensorflowRecords(datasetPath + os.sep + "annotations",
                                                     datasetPath + os.sep + "images")
    def organize(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass
    def createModel(self):
        pass

