from objectDetectors.YOLOObjectDetector import darknetDetector
import annotationParser
import os
import wget
import objectDetectors.YOLOObjectDetector.functions as fn


from imutils import paths
import shutil



class TinyYoloV3Detector(darknetDetector.DarknetAbstract):
    def __init__(self):
        darknetDetector.DarknetAbstract.__init__(self)
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
        darknetDetector.DarknetAbstract.organize(dataset_path,output_path,train_percentage)

    def createModel(self, dataset_path):
        file = open(os.path.join(dataset_path, "classes.names"))
        classes = []
        for line in file:
            classes.append(line)
        n_classes = fn.contarClases(classes)
        dataset_name = dataset_path[dataset_path.rfind(os.sep)+1:]
        fn.generaFicheroData(dataset_path,n_classes,dataset_name)
        fn.generaFicherosTinyYoloTrain(dataset_path, dataset_name, n_classes)

    def train(self, dataset_path, darknet_path):
        # !git clone https: // github.com / AlexeyAB / darknet.git
        # !make
        # !./darknet detector train voc.data yolov3-voc.cfg darknet53.conv.74
        data = [p for p in os.listdir(dataset_path) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(dataset_path) if p.endswith(".cfg")][0]
        # if (not os.path.exists("./darknet")):
        #     os.system("git clone https://github.com/AlexeyAB/darknet.git")
        #
        #     # probar mejor lo de la ruta para hacer la compilacion del proyecto
        #     os.system("make ./darknet/Makefile")

        # os.system("./darknet/darknet detector train /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012dataset.data /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012datasettrain.cfg darknet53.conv.74")
        os.system(os.path.join(darknet_path, "darknet") + " detector train " + os.path.abspath(dataset_path+ os.sep + data) + " " +
                  os.path.abspath(dataset_path+ os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")

    def evaluate(self, dataset_path, darknet_path):
        data = [p for p in os.listdir(dataset_path) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(dataset_path) if p.endswith(".cfg")][0]
        os.system(os.path.join(darknet_path, "darknet") + " detector map " + os.path.abspath(dataset_path + os.sep + data) + " " + os.path.abspath(
                dataset_path + os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")

def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
