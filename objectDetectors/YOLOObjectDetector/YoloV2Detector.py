from objectDetectors.YOLOObjectDetector import darknetDetector
import annotationParser
import os
import wget
import objectDetectors.YOLOObjectDetector.funciones as fn

class YoloV2Detector(darknetDetector.DarknetAbstract):
    def __init__(self):
        darknetDetector.DarknetAbstract.__init__(self)
        self.url = "https://pjreddie.com/media/files/yolov3.weights"
        self.urlWeights = "https://pjreddie.com/media/files/darknet53.conv.74"

        # filename = wget.download(self.url)
        # filename2 = wget.download(self.urlWeights)




    # Este metodo transformara si hace falta del formato de anotacion que tenga a pascalvoc para poder trabajar con el
    def transform(self, datasetPath):
        annotationParser.PascalVOC2TensorflowRecords(datasetPath + os.sep + "annotations",
                                                     datasetPath + os.sep + "images")
    def organize(self):

        # mover los ficheros para crear conjunto de entrenamiento y test
        # para yolo carpeta donde tienes una carpeta que se llame train y otra test.
        # train dentro JPEGIMAGES y otra de labels
        # test igual
        # fichero data
        # lista con el path de las imagenes a la carpeta



        # para el parser no crear tambien la carpeta de las imagenes sino que crear solo las anotaciones y luego aqui
        # le tendremos que poner la ruta de esas anotaciones y que se organice todo como quiere el framework en cada caso



        pass
    def train(self):
        # entrenar solo el modelo
        pass
    def evaluate(self):
        # evaluar el modelo en el conjunto de test
        pass
    def createModel(self, datasetPath):
        print(os.path.join(datasetPath, "classes.name"))
        print("____________________________________________")
        file = open(os.path.join(datasetPath, "classes.name"))
        classes = []
        for linea in file:
            classes.append(linea)
        n_classes = len(classes)
        annotationParser.classes = classes
        datasetName = datasetPath[datasetPath.rfind(os.sep):]
        print(datasetName)
        print("___________________________________________________________")
        fn.generaFicheroData("config/file.data",n_classes,datasetName)

        # fn.generaFicherosYoloTrain()


def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
