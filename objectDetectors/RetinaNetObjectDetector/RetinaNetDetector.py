from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.RetinaNetObjectDetector import functions as fn
from Predictors.RetinanetPredict import RetinanetPredictor
from Evaluators.MapEvaluator import MapEvaluator as Map
import  os
import shutil
from imutils import paths

class RetinaNetDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name):
        super(RetinaNetDetector, self).__init__(dataset_path, dataset_name)
        self.model = "retinanet"
        # IObjectDetection.__init__(self, dataset_path, dataset_name)
        # if (not os.path.exists("./keras-retinanet")):
            # os.system("git clone https://github.com/fizyr/keras-retinanet")
        # os.system('sudo python3 keras-retinanet/setup.py install')

    def transform(self):
        # listaFicheros = list(paths.list_files(pathImages, validExts=(".jpg")))
        train_list = list(paths.list_files(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train"), validExts=(".jpg")))
        test_list = list(paths.list_files(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test"), validExts=(".jpg")))
        annotations_dir_train = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "train", "Annotations")
        annotations_dir_test = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "test", "Annotations")

        # train_list, test_list, _, _ = train_test_split(listaFicheros, listaFicheros, train_size=porcentaje,random_state=5)
        # creamos la estructura de carpetas, la primera contendra las imagenes del entrenamiento
        if (not os.path.exists(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, 'images'))):
            os.makedirs(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, 'images'))
        # esta carpeta contendra las anotaciones de las imagenes de entrenamiento
        # os.makedirs(os.path.join(output_path, Nproyecto, 'annotations'))

        traincsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_train.csv"), "w")
        testcsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_test.csv"), "w")
        classes = set()
        for file in train_list:
            name = os.path.basename(file).split('.')[0]
            # annotation_file = file[:file.rfind(os.sep) + 1] + name + ".xml"
            annotation_file = os.path.join(annotations_dir_train, name + ".xml")

            image_splited = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, 'images', name + '.jpg')
            boxes = fn.obtain_box(annotation_file)
            for bo in boxes:
                traincsv.write(
                    os.path.abspath(image_splited) + "," + str(int(bo["x_min"])) + "," + str(int(bo["y_min"])) + ","
                    + str(int(bo["x_max"])) + "," + str(int(bo["y_max"])) + "," + str(bo["class"]) + "\n")
                classes.add(bo["class"])
            shutil.copy(file, image_splited)
        # para las imagenes de entrenamiento
        for file in test_list:
            name = os.path.basename(file).split('.')[0]
            # annotation_file = file[:file.rfind(os.sep) + 1] + name + ".xml"
            annotation_file = os.path.join(annotations_dir_test, name + ".xml")

            image_splited = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, 'images', name + '.jpg')
            boxes = fn.obtain_box(annotation_file)
            for bo in boxes:
                testcsv.write(
                    os.path.abspath(image_splited) + "," + str(int(bo["x_min"])) + "," + str(int(bo["y_min"])) + ","
                    + str(int(bo["x_max"])) + "," + str(int(bo["y_max"])) + "," + str(bo["class"]) + "\n")
                classes.add(bo["class"])
            shutil.copy(file, image_splited)

        classescsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_classes.csv"), "w")
        rows = [",".join([c, str(i)]) for (i, c) in enumerate(classes)]
        classescsv.write("\n".join(rows))
        classescsv.close()
        traincsv.close()
        testcsv.close()
        # shutil.rmtree(os.path.join(output_path, Nproyecto, "train"))
        # shutil.rmtree(os.path.join(output_path, Nproyecto, "test"))

    # def organize(self, train_percentage):
    #     super(RetinaNetDetector, self).organize(train_percentage)
        # IObjectDetection.organize(self, train_percentage)

        # dataset_name = self.DATASET[self.DATASET.rfind(os.sep) + 1:]
        # images_path = list(paths.list_files(datasetPath, validExts=(".jpg")))
        # annotations_path = list(paths.list_files(datasetPath, validExts=(".xml")))

    # def createModel(self):
    #     pass

    def train(self, framework_path = None, n_gpus = 1):
        # dataset_name = self.DATASET[self.DATASET.rfind(os.sep)+1:]
        epochs = 5
        batch_size = 2
        # Como en todos los casos anteriores el dataset debe estar guardado ahi ya dividido en train/test
        traincsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_train.csv"))
        num_files = len(traincsv.readlines())
        steps = round(num_files/batch_size)
        command = "python3 " + framework_path + "/bin/train.py --batch-size 2 --steps " + str(steps) + " --epochs " + str(epochs) + " --snapshot-path " +\
                  os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "models") + " csv " + os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_train.csv ") +  \
                  os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_classes.csv")
        if not os.path.exists(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models")):
            os.mkdir(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models"))
        os.system(command)
        os.system(framework_path + '/retinanet -convert-model datasets/snapshots_' + self.DATASET_NAME + '/resnet50_csv_5.h5 datasets/snapshots_' + self.DATASET_NAME + '/output.h5')

        shutil.rmtree(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, 'images'))
        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME+"_classes.csv"))
        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME+"_test.csv"))
        os.remove(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME+"_train.csv"))

        # retinanet-train --batch-size 2 --steps 1309 --epochs 50 --weights weapons/resnet50_coco_best_v2.1.0.h5 --snapshot-path weapons/snapshots csv weapons/retinanet_train.csv
        # weapons/retinanet_classes.csv
        #
        # retinanet -convert-model weapons/snapshots/resnet50_csv_50.h5 output.h5
        #

    def evaluate(self):
        # yoloPredict = DarknetPredict(imagePaths,modelWeights,classesFile,modelConfiguration)
        yoloPredict = RetinanetPredictor(
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "resnet50_csv_5.h5"),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"))
        map = Map(yoloPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()