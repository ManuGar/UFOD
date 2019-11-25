import os
from imutils import paths

from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.YOLOObjectDetector import functions as fn
import wget

class DarknetAbstract(IObjectDetection):
    def __init__(self,dataset_path, dataset_name):
        super(DarknetAbstract, self).__init__(dataset_path, dataset_name)
    def transform(self):
        # this function prepare the dataset to the yolo estructure
        f = open(os.path.join(self.DATASET, "classes.names"), "r")
        for line in f:
            fn.CLASSES.append(line.split("\n")[0])
        train_images = list(
            paths.list_files(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train"), validExts=(".jpg")))
        test_images = list(
            paths.list_files(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test"), validExts=(".jpg")))
        traintxt = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train.txt"), "w")
        testtxt = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test.txt"),"w")
        for tr_im in train_images:
            traintxt.write(os.path.abspath(tr_im) + "\n")
        for te_im in test_images:
            testtxt.write(os.path.abspath(te_im) + "\n")
        traintxt.close()
        testtxt.close()
        fn.PascalVOC2YOLO(self.OUTPUT_PATH, self.DATASET_NAME)  # , datasetPath + os.sep + "images"
    # def organize(self, train_percentage):
    #     super(DarknetAbstract, self).organize(train_percentage)
    # def createModel(self):
    #     pass
    def train(self, framework_path = None, n_gpus = 1):
        data = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) if p.endswith(".data")][0]
        confi = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) if p.endswith(".cfg")][0]
        if not os.path.exists("objectDetectors/YOLOObjectDetector/darknet53.conv.74"):
            wget.download("https://www.dropbox.com/s/67dvod7i509lmd8/darknet53.conv.74?dl=1",
                          "objectDetectors/YOLOObjectDetector/darknet53.conv.74")
        # os.system("./darknet/darknet detector train /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012dataset.data /home/magarcd/Escritorio/salida3/VOC2012dataset/VOC2012datasettrain.cfg darknet53.conv.74")
        if not os.path.exists(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models")):
            os.mkdir(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models"))
        os.system(os.path.join(framework_path, "darknet") + " detector train " + os.path.abspath(
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, data)) + " " +
                  os.path.abspath(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME,
                                               confi)) + " objectDetectors/YOLOObjectDetector/darknet53.conv.74 -dont_show -gpus " + ",".join(
            str(i) for i in range(0, n_gpus)))


        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train.txt"))
        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test.txt"))
        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train.txt"))
        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + ".data"))



    # def evaluate(self, framework_path = None):
    #     pass