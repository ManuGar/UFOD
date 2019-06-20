from objectDetectors.YOLOObjectDetector import darknetDetector
import annotationParser
import os
import wget
import objectDetectors.YOLOObjectDetector.funciones as fn




















# Es una copia del v3, mirar que cambios hay que hacer en crearFicherosYoloTrain de funciones para que valgan para v2.
# Igual lo que hay que hacer es hacer copias o algo similar para que en cada modelo se cree el correspondiente fichero
# de configuracion. Algo como tener el mensaje en cada clase y se llama a la funcion que sea la que cree todo.
# De hecho entre el entrenamiento y el test tampoco hay demasidadas diferencias en cuanto a ese fichero.
class YoloV2Detector(darknetDetector.DarknetAbstract):
    def __init__(self):
        darknetDetector.DarknetAbstract.__init__(self)
        url = "https://pjreddie.com/media/files/yolov2.weights"
        urlWeights = "https://pjreddie.com/media/files/darknet19_448.conv.23"
        if(not (os.path.isfile("objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "yolov2.weights"))):
            filename = wget.download(url, "objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "yolov3.weights")
        if(not (os.path.isfile("objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74"))):
            filename2 = wget.download(urlWeights,"objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "d")




    # Este metodo transformara si hace falta del formato de anotacion que tenga a pascalvoc para poder trabajar con el
    def transform(self, datasetPath):
        annotationParser.PascalVOC2YOLO(datasetPath + os.sep + "annotations")#, datasetPath + os.sep + "images"
    def organize(self, datasetPath):

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
        # para entrenar solo veo que se hagan comandos desde consola no veo que nadie lo ejecute desde el propio python
        # hay una opcion desde opencv de cargar los pesos y el archivo de configuracion para cargar la red. Igual con esa se puede hacer
        pass
    def evaluate(self):
        # evaluar el modelo en el conjunto de test
        pass




    def createModel(self, datasetPath):
        file = open(os.path.join(datasetPath, "classes.name"))
        classes = []
        for linea in file:
            classes.append(linea)
        n_classes = fn.contarClases(classes)
        annotationParser.classes = classes
        datasetName = datasetPath[datasetPath.rfind(os.sep)+1:]
        # os.path.join(darknetPath, "cfg", Nproyecto + ".data")
        fn.generaFicheroData(datasetPath,n_classes,datasetName)
        fn.generaFicherosYoloTrain(datasetPath, datasetName, n_classes)


def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
