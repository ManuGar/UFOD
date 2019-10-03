from objectDetectors.YOLOObjectDetector import darknetDetector
import annotationParser
import os
import wget
import objectDetectors.YOLOObjectDetector.functions as fn

import subprocess
import shutil

class YoloV3Detector(darknetDetector.DarknetAbstract):
    def __init__(self):
        darknetDetector.DarknetAbstract.__init__(self)

        # if (not os.path.exists("./darknet")):
        #     os.system("git clone https://github.com/pjreddie/darknet")
        # darknetDetector.DarknetAbstract.__init__(self)
        # urlWeights = "https://pjreddie.com/media/files/darknet53.conv.74"
        # if(not (os.path.isfile("objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74"))):
        #     filename2 = wget.download(urlWeights,"objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74")

    def transform(self, dataset_path, output_path):
        darknetDetector.DarknetAbstract.transform(self, dataset_path, output_path)
        # f = open(classes_path, "r")
        # if (not os.path.exists(output_path)):
        #     os.makedirs(output_path)
        # classes_name = open(os.path.join(output_path,"classes.names"), "w")
        # for line in f:
        #     annotationParser.CLASSES.append(line.split("\n")[0])
        #     classes_name.write(line)
        # annotationParser.PascalVOC2YOLO(dataset_path, output_path)#, datasetPath + os.sep + "images"

    def organize(self, dataset_path, output_path, train_percentage):
        darknetDetector.DarknetAbstract.organize(dataset_path, output_path, train_percentage)

    def createModel(self, dataset_path):
        # en el fichero de data hay que asegurarse de la ruta que se le pone, por que le aniade data/ que no se ha creado esa carpeta
        file = open(os.path.join(dataset_path, "classes.names"))
        classes = []
        for line in file:
            classes.append(line)
        n_classes = fn.contarClases(classes)
        annotationParser.CLASSES = classes
        dataset_name = dataset_path[dataset_path.rfind(os.sep)+1:]
        # os.path.join(darknetPath, "cfg", Nproyecto + ".data")
        fn.generaFicheroData(dataset_path,n_classes,dataset_name)
        fn.generaFicherosYoloTrain(dataset_path, dataset_name, n_classes)

    def train(self, dataset_path, darknet_path):
        # !git clone https: // github.com / AlexeyAB / darknet.git
        # !make
        # !./darknet detector train voc.data yolov3-voc.cfg darknet53.conv.74
        data = [p for p in os.listdir(dataset_path) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(dataset_path) if p.endswith(".cfg")][0]


        # os.system("make ./darknet/Makefile")
        # os.system("./darknet/darknet detector train /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012dataset.data /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012datasettrain.cfg darknet53.conv.74")
        os.system(os.path.join(darknet_path, "darknet") + " detector train " + os.path.abspath(dataset_path+ os.sep + data) + " " +
                  os.path.abspath(dataset_path+ os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")


        # para entrenarlo pasar solo el path que contiene todo, ahi tenemos el datset dividido y cogemos la parte de
        # como se van a usar todos los pasos seguidos no va a hacer falta que se le pasen los parametros de uno en uno
        # se le pasa el path del dataset y como sabemos perfectamente la estructura que va a tener se buscan a partir del
        # del dataset que le hemos pasado
        # tambien habria que compilar darknet para que se pudiera usar o eso lo suponemos que esta hecho de antes

    def evaluate(self, dataset_path, darknet_path):
        # evaluar el modelo en el conjunto de test

        data = [p for p in os.listdir(dataset_path) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(dataset_path) if p.endswith(".cfg")][0]

        os.system(os.path.join(darknet_path, "darknet") + " detector map " + os.path.abspath(dataset_path + os.sep + data) + " " + os.path.abspath(
                dataset_path + os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")
        # print("./darknet/darknet detector map " + os.path.abspath(dataset_path + os.sep + data) + " " + os.path.abspath(
        #         dataset_path + os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")


def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
