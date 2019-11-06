class IPredictor(object):
    def __init__(self, imagePaths,modelWeights,classesFile):
        self.imagePaths = imagePaths
        self.modelWeights = modelWeights
        self.classesFile = classesFile
        pass
    def predict(self):
        pass
