from objectDetectionInterface import IObjectDetection

class TensorflowAbstract(IObjectDetection):
    def __init__(self):
        pass



    # Este metodo transformara si hace falta del formato de anotacion que tenga a pascalvoc para poder trabajar con el
    def transform(self):
        pass
    def organize(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass
    def createModel(self):
        pass
