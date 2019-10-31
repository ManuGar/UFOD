from objectDetectors.YOLOObjectDetector import darknetDetector
import os
import wget
import objectDetectors.YOLOObjectDetector.functions as fn


class YoloV3Detector(darknetDetector.DarknetAbstract):
    def __init__(self, dataset_path, dataset_name):
        darknetDetector.DarknetAbstract.__init__(self,dataset_path,dataset_name)
        # if (not os.path.exists("./darknet")):
        #     os.system("git clone https://github.com/pjreddie/darknet")
        # darknetDetector.DarknetAbstract.__init__(self)
        # urlWeights = "https://pjreddie.com/media/files/darknet53.conv.74"
        # if(not (os.path.isfile("objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74"))):
        #     filename2 = wget.download(urlWeights,"objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74")

    # def transform(self):
    #     super(YoloV3Detector, self).transform()
    #     # darknetDetector.DarknetAbstract.transform(self)
    # def organize(self, train_percentage):
    #     super(YoloV3Detector, self).organize(train_percentage)
    #     # darknetDetector.DarknetAbstract.organize(self, train_percentage)

    def createModel(self):
        file = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"))
        classes = []
        for line in file:
            classes.append(line)
        n_classes = fn.contarClases(classes)
        fn.CLASSES = classes
        # dataset_name = dataset_path[dataset_path.rfind(os.sep)+1:]
        # os.path.join(darknetPath, "cfg", Nproyecto + ".data")
        fn.generaFicheroData(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME),n_classes,self.DATASET_NAME)
        fn.generaFicherosYoloTrain(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME), self.DATASET_NAME, n_classes)

    def train(self, framework_path = None):
        data = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) if p.endswith(".cfg")][0]
        if not os.path.exists("objectDetectors/YOLOObjectDetector/darknet53.conv.74"):
            wget.download("https://www.dropbox.com/s/67dvod7i509lmd8/darknet53.conv.74?dl=0", "objectDetectors/YOLOObjectDetector/darknet53.conv.74")

        os.system(os.path.join(framework_path, "darknet") + " detector train " + os.path.abspath(self.DATASET+ os.sep + data) + " " +
                  os.path.abspath(self.DATASET+ os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")

        # para entrenarlo pasar solo el path que contiene todo, ahi tenemos el datset dividido y cogemos la parte de
        # como se van a usar todos los pasos seguidos no va a hacer falta que se le pasen los parametros de uno en uno
        # se le pasa el path del dataset y como sabemos perfectamente la estructura que va a tener se buscan a partir del
        # del dataset que le hemos pasado
        # tambien habria que compilar darknet para que se pudiera usar o eso lo suponemos que esta hecho de antes

    def evaluate(self, framework_path = None):
        # evaluar el modelo en el conjunto de test

        data = [p for p in os.listdir(self.DATASET) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(self.DATASET) if p.endswith(".cfg")][0]

        os.system(os.path.join(framework_path, "darknet") + " detector map " + os.path.abspath(self.DATASET + os.sep + data) + " " + os.path.abspath(
                self.DATASET + os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74 -dont_show")
        # print("./darknet/darknet detector map " + os.path.abspath(dataset_path + os.sep + data) + " " + os.path.abspath(
        #         dataset_path + os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")

def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
