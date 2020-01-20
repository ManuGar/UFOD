class IPredictor(object):
    def __init__(self,modelWeights,classesFile):
        self.modelWeights = modelWeights
        self.classesFile = classesFile
    def predict(self, imagePaths):
        pass
    def predictImage(self, imagePath):
        pass
