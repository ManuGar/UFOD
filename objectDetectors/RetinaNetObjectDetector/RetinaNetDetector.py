from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.RetinaNetObjectDetector import functions as fn
import  os

class RetinaNetDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name):
        IObjectDetection.__init__(self, dataset_path, dataset_name)
        # if (not os.path.exists("./keras-retinanet")):
            # os.system("git clone https://github.com/fizyr/keras-retinanet")
        # os.system('sudo python3 keras-retinanet/setup.py install')

    def transform(self):
        fn.datasetSplit(self.DATASET_NAME,self.OUTPUT_PATH)

    def organize(self, train_percentage):
        IObjectDetection.organize(self, train_percentage)
        # dataset_name = self.DATASET[self.DATASET.rfind(os.sep) + 1:]
        # images_path = list(paths.list_files(datasetPath, validExts=(".jpg")))
        # annotations_path = list(paths.list_files(datasetPath, validExts=(".xml")))

    def createModel(self):
        pass

    def train(self, framework_path = None):
        # dataset_name = self.DATASET[self.DATASET.rfind(os.sep)+1:]
        epochs = 50
        batch_size = 2
        # Como en todos los casos anteriores el dataset debe estar guardado ahi ya dividido en train/test
        traincsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME + "_train.csv"))
        num_files = len(traincsv.readlines())
        steps = round(num_files/batch_size)
        command = framework_path + "/retinanet-train --batch-size 2 --steps " + str(steps) + " --epochs " + str(epochs) + " --snapshot-path " +\
                  self.OUTPUT_PATH + "/snapshots" + " csv " + self.OUTPUT_PATH + os.sep + self.DATASET_NAME + "_train.csv " +  \
                  self.OUTPUT_PATH + os.sep + self.DATASET_NAME + "_classes.csv"
        os.system(command)
        os.system(framework_path + '/retinanet -convert-model weapons/snapshots/resnet50_csv_50.h5 output.h5')


        # retinanet-train --batch-size 2 --steps 1309 --epochs 50 --weights weapons/resnet50_coco_best_v2.1.0.h5 --snapshot-path weapons/snapshots csv weapons/retinanet_train.csv
        # weapons/retinanet_classes.csv
        #
        # retinanet -convert-model weapons/snapshots/resnet50_csv_50.h5 output.h5
        #

    def evaluate(self, framework_path = None):
        os.system("retinanet-evaluate csv " + self.OUTPUT_PATH + os.sep + self.OUTPUT_PATH + "_train.csv " +  self.OUTPUT_PATH +
                  os.sep + self.DATASET_NAME + "_classes.csv " + self.OUTPUT_PATH + os.sep+ " output.h5")
        # retinanet-evaluate csv weapons/retinanet_test.csv weapons/retinanet_classes.csv output.h5