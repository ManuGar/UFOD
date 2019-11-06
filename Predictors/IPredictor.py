class IPredictor(object):
    def __init__(self,modelWeights,classesFile):
        self.modelWeights = modelWeights
        self.classesFile = classesFile
        pass
    def predict(self, imagePaths):
        pass
