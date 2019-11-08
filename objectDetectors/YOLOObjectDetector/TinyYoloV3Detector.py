from objectDetectors.YOLOObjectDetector.darknetDetector import DarknetAbstract
from Predictors.DarknetPredict import DarknetPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
import os
import wget
import objectDetectors.YOLOObjectDetector.functions as fn



class TinyYoloV3Detector(DarknetAbstract):
    def __init__(self, dataset_path, dataset_name):
        DarknetAbstract.__init__(self,dataset_path,dataset_name)
        self.model = "tinyYolov3"
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
        file = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"))
        classes = []
        for line in file:
            classes.append(line)
        n_classes = fn.contarClases(classes)
        fn.generaFicheroData(self.OUTPUT_PATH,n_classes,self.DATASET_NAME)
        fn.generaFicherosTinyYoloTrain(self.OUTPUT_PATH, self.DATASET_NAME, n_classes)

    def train(self, framework_path = None, n_gpus = 1):
        data = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) if p.endswith(".cfg")][0]
        if not os.path.exists("objectDetectors/YOLOObjectDetector/darknet53.conv.74"):
            wget.download("https://www.dropbox.com/s/67dvod7i509lmd8/darknet53.conv.74?dl=1", "objectDetectors/YOLOObjectDetector/darknet53.conv.74")
        # os.system("./darknet/darknet detector train /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012dataset.data /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012datasettrain.cfg darknet53.conv.74")
        os.system(os.path.join(framework_path, "darknet") + " detector train " + os.path.abspath(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, data)) + " " +
                  os.path.abspath(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME, confi)) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74 -dont_show -gpus " + ",".join(str(i) for i in range(0,n_gpus)) )

    def evaluate(self, framework_path = None):
        tinyyoloPredict = DarknetPredict(
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "TinyTrain_final.weights"),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "TinyTrain.cfg"))
        map = Map(tinyyoloPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME),self.model)
        map.evaluate()

def main():
    # PascalVOC2TensorflowRecords("../datasets/VOC2012/Annotations", "../datasets/VOC2012/JPEGImages")
    pass

if __name__ == "__main__":
    main()
