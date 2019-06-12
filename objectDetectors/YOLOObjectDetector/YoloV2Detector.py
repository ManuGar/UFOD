from objectDetectors.YOLOObjectDetector import darknetDetector
import annotationParser
import os
import wget
import objectDetectors.YOLOObjectDetector.funciones as fn

class YoloV2Detector(darknetDetector):
    def __init__(self):
        darknetDetector.__init__()
        url = "https://pjreddie.com/media/files/yolov3.weights"
        urlWeights = "https://pjreddie.com/media/files/darknet53.conv.74"
        filename = wget.download(url)
        filename2 = wget.download(urlWeights)




    # Este metodo transformara si hace falta del formato de anotacion que tenga a pascalvoc para poder trabajar con el
    def transform(datasetPath):
        annotationParser.PascalVOC2TensorflowRecords(datasetPath + os.sep + "annotations",
                                                     datasetPath + os.sep + "images")
    def organize(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass
    def createModel(datasetPath):
        file = open(os.join(datasetPath, "classes.name"))
        classes = []
        for linea in file:
            classes.append(linea)
        n_classes = len(classes)
        annotationParser.classes = classes
        datasetName = datasetPath[datasetPath.rfind(os.sep):]
        fn.generaFicheroData("config/file.data",n_classes,datasetName)

        # fn.generaFicherosYoloTrain()