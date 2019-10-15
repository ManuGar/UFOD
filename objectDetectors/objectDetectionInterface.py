import os

class IObjectDetection(object):

    def __init__(self, dataset_path, dataset_name):
        self.DATASET =dataset_path
        self.DATASET_NAME =dataset_name
        self.OUTPUT_PATH = os.path.join(".","datasets")
        pass
    def transform(self):

        pass
    def organize(self, train_percentage):
        pass
    def createModel(self):
        pass
    def train(self, framework_path = None):
        pass
    def evaluate(self, framework_path = None):
        pass