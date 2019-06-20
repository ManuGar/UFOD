from objectDetectors.objectDetectionInterface import IObjectDetection
class DarknetAbstract(IObjectDetection):
    def __init__(self):
        IObjectDetection.__init__(self)
        pass
    def transform(self, dataset_path, classes_path, output_path):
        pass
    def organize(self, datasetPath, output_path, train_percentage):
        pass
    def createModel(self, datasetPath):
        pass
    def train(self, dataset_path):
        pass
    def evaluate(self, dataset_path):
        pass