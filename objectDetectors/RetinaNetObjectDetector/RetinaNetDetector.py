from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.RetinaNetObjectDetector import functions as fn
from imutils import paths
import  os

class RetinaNetDetector(IObjectDetection):
    def __init__(self):
        IObjectDetection.__init__(self)

        # if (not os.path.exists("./keras-retinanet")):
            # os.system("git clone https://github.com/fizyr/keras-retinanet")

        # os.system('sudo python3 keras-retinanet/setup.py install')


    def transform(self, dataset_path, classes_path, output_path):
        pass

    def organize(self, datasetPath, output_path, train_percentage):
        dataset_name = datasetPath[datasetPath.rfind(os.sep) + 1:]
        # images_path = list(paths.list_files(datasetPath, validExts=(".jpg")))
        # annotations_path = list(paths.list_files(datasetPath, validExts=(".xml")))
        fn.datasetSplit(dataset_name,output_path,datasetPath,train_percentage)



    def createModel(self, datasetPath):
        pass

    def train(self, dataset_path):
        dataset_name = dataset_path[dataset_path.rfind(os.sep)+1:]
        epochs = 50
        batch_size = 2
        traincsv = open(os.path.join(dataset_path, dataset_name + "_train.csv"))
        num_files = len(traincsv.readlines())
        steps = round(num_files/batch_size)
        command = "retinanet-train --batch-size 2 --steps " + str(steps) + " --epochs " + str(epochs) + " --snapshot-path " +\
                  dataset_path + "/snapshots" + " csv " + dataset_path + os.sep + dataset_name + "_train.csv " +  \
                  dataset_path + os.sep + dataset_name + "_classes.csv"
        os.system(command)
        os.system('retinanet -convert-model weapons/snapshots/resnet50_csv_50.h5 output.h5')


        # retinanet-train --batch-size 2 --steps 1309 --epochs 50 --weights weapons/resnet50_coco_best_v2.1.0.h5 --snapshot-path weapons/snapshots csv weapons/retinanet_train.csv
        # weapons/retinanet_classes.csv
        #
        # retinanet -convert-model weapons/snapshots/resnet50_csv_50.h5 output.h5
        #

    def evaluate(self, dataset_path):
        dataset_name = dataset_path[dataset_path.rfind(os.sep)+1:]
        os.system("retinanet-evaluate csv " + dataset_path + os.sep + dataset_name + "_train.csv " +  dataset_path +
                  os.sep + dataset_name + "_classes.csv " + dataset_path + os.sep+ " output.h5")
        # retinanet-evaluate csv weapons/retinanet_test.csv weapons/retinanet_classes.csv output.h5