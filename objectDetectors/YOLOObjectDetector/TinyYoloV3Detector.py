from objectDetectors.YOLOObjectDetector import darknetDetector
import os
import wget
import objectDetectors.YOLOObjectDetector.functions as fn



class TinyYoloV3Detector(darknetDetector.DarknetAbstract):
    def __init__(self, dataset_path, dataset_name):
        darknetDetector.DarknetAbstract.__init__(self,dataset_path,dataset_name)
        # urlWeights = "https://pjreddie.com/media/files/darknet53.conv.74"
        # if(not (os.path.isfile("objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74"))):
        #     filename2 = wget.download(urlWeights,"objectDetectors" + os.sep + "YOLOObjectDetector" + os.sep + "darknet53.conv.74")

    # def transform(self):
    #     super(TinyYoloV3Detector, self).transform()
    #     # darknetDetector.DarknetAbstract.transform(self)


    # def organize(self, train_percentage):
    #     super(TinyYoloV3Detector, self).organize(train_percentage)
    #     # darknetDetector.DarknetAbstract.organize(self, train_percentage)

    def createModel(self):
        file = open(os.path.join(self.OUTPUT_PATH, "classes.names"))
        classes = []
        for line in file:
            classes.append(line)
        n_classes = fn.contarClases(classes)
        fn.generaFicheroData(self.OUTPUT_PATH,n_classes,self.DATASET_NAME)
        fn.generaFicherosTinyYoloTrain(self.OUTPUT_PATH, self.DATASET_NAME, n_classes)

    def train(self, framework_path = None):
        data = [p for p in os.listdir(self.DATASET) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(self.DATASET) if p.endswith(".cfg")][0]
        if not os.path.exists("objectDetectors/YOLOObjectDetector/darknet53.conv.74"):
            wget.download("https://www.dropbox.com/s/67dvod7i509lmd8/darknet53.conv.74?dl=0", "objectDetectors/YOLOObjectDetector/darknet53.conv.74")
        # os.system("./darknet/darknet detector train /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012dataset.data /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012datasettrain.cfg darknet53.conv.74")
        os.system(os.path.join(framework_path, "darknet") + " detector train " + os.path.abspath(self.OUTPUT_PATH+ os.sep + data) + " " +
                  os.path.abspath(framework_path + os.sep + confi) + "objectDetectors/YOLOObjectDetector/darknet53.conv.74 -dont_show")

    def evaluate(self, framework_path = None):
        data = [p for p in os.listdir(framework_path) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(framework_path) if p.endswith(".cfg")][0]
        os.system(os.path.join(framework_path, "darknet") + " detector map " + os.path.abspath(self.DATASET + os.sep + data) + " " + os.path.abspath(
                self.DATASET + os.sep + confi) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74")

def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
