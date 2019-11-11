
import os

from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.TensorflowObjectDetector.functions import PascalVOC2TensorflowRecords
from objectDetectors.TensorflowObjectDetector import functions as fn

import wget
import tarfile

import shutil

class TensorflowDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name, model):
        super(TensorflowDetector, self).__init__(dataset_path,dataset_name)
        self.model = model
        # IObjectDetection.__init__(self, dataset_path, dataset_name)

    def transform(self):
        # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        if (not os.path.exists(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME))):
            os.makedirs(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME))
        class_path = os.path.join(self.DATASET, "classes.names")
        result_path = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)
        file = open(class_path, "r")
        cl_txt = ""
        i = 1
        for cl in file:
            cl_txt += "item {\n\tid: " + str(i) + "\n\tname: '" + cl.split("\n")[0] + "'\n}\n"
            i += 1
        label_map = open(os.path.join(result_path, "label_map.pbtxt"), "w+")
        label_map.write(cl_txt)
        label_map.close()
        print(cl_txt)

        PascalVOC2TensorflowRecords(self.DATASET, self.OUTPUT_PATH)
        # shutil.rmtree(dataset_path)


    # En este caso tendremos que hacer primero la division del dataset para que luego la transformacion se haga
    # desde el dataset de entrenamiento y de evaluacion
    def organize(self, train_percentage):
        # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        fn.datasetSplit(self.DATASET_NAME,self.DATASET,self.OUTPUT_PATH,train_percentage)

    def createModel(self):

        # aqui ademas de crear el archivo de configuracion hay que descargar el modelo que se va a usar en ese modelo de configuracion
        # ademas, hay que usar el modelo ssd_inception_v2_coco y el faster_rcnn_resnet50_coco
        # crear como siempre una clase para cada "modelo" que hay. Solo habria que crear estos dos para tensorflow de momento. con el organize y el transform igual.
        # lo unico que cambiaria es el create model, y hay dudas de que haya que cambiar el train por lo que tambien podria ir en el padre (que deberia ser esta clase)
        aux_path = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "pre-trained-model", self.model + "_2018_01_28.tar.gz")
        if (not os.path.exists(os.path.join(self.OUTPUT_PATH, "pre-trained-model"))):
            os.makedirs(os.path.join(self.OUTPUT_PATH, "pre-trained-model"))
            model = wget.download("http://download.tensorflow.org/models/object_detection/" + self.model + "_2018_01_28.tar.gz",
                aux_path)
            # model = wget.download( "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz",
            #     os.path.join(self.DATASET, "pre-trained-model", "ssd_inception_v2_coco.tar.gz"))
            # tar = tarfile.open(os.path.join(self.DATASET, "pre-trained-model", "ssd_inception_v2_coco.tar.gz"))
            tar = tarfile.open(aux_path)
            tar.extractall(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "pre-trained-model"))
            tar.close()
            os.remove(aux_path)

        classes = []
        n_classes = 0

        with open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "label_map.pbtxt")) as f:
            for line in f:
                classes.append(line)
                if line.__contains__("item {"):
                    n_classes += 1

        fn.generateTensorFlowConfigFile(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME),n_classes)


    def train(self, framework_path= None, n_gpus = 1):
        # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        # os.system("python3 " +os.path.join(tensorFlow_path, "research", "object_detection", "legacy", "train.py") + " --logtostderr --train_dir=" + os.path.abspath(
        #     dataset_path) + " --pipeline_config_path=" + os.path.join(dataset_path, "ssd_inception_v2_pets.config"))

        os.system("python3 " + os.path.join(framework_path, "research", "object_detection", "legacy",
                                            "train.py") + " --logtostderr --train_dir=" + os.path.abspath(
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)) + " --pipeline_config_path=" + os.path.join(self.OUTPUT_PATH, self.DATASET_NAME + ".config"))

        # python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

    def evaluate(self, framework_path = None):
        pass