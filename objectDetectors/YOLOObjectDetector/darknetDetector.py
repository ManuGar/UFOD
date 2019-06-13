from objectDetectors.objectDetectionInterface import IObjectDetection
class DarknetAbstract(IObjectDetection):
    def __init__(self):
        IObjectDetection.__init__(self)
        pass
    def transform(self, datasetPath):
        pass
    def organize(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass
    def createModel(self, datasetPath):
        pass
