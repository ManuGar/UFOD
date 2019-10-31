from objectDetectors.objectDetectionInterface import IObjectDetection
from imutils import paths
import gluoncv as gcv
import shutil
import os


# from objectDetectors.MXNetObjectDetector import  functions as fn

# from gluoncv import model_zoo, data, utils
# from mxnet import gluon, autograd
# from mxnet.gluon.model_zoo import vision as models
# from mxnet.gluon import data, utils
# from mxnet.gluon.data.vision import datasets, transforms
# import mxnet as mx




class MxNetDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name):
        IObjectDetection.__init__(self, dataset_path, dataset_name)

    def transform(self):
        listaFicheros_train = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train"), validExts=(".jpg")))
        listaFicheros_test = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"test"), validExts=(".jpg")))

        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME)

        shutil.copytree(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","JPEGImages"), os.path.join(outputPath, "JPEGImages"))
        shutil.copytree(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","Annotations"), os.path.join(outputPath, "Annotations"))
        if (not (os.path.exists(os.path.join(outputPath, "ImageSets")))):
            os.makedirs(os.path.join(outputPath, "ImageSets", "Main"))

        shutil.copy(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"), outputPath)
        traintxt = open(os.path.join(outputPath, "ImageSets", "Main", "train.txt"), "w")
        testtxt = open(os.path.join(outputPath, "ImageSets", "Main", "test.txt"), "w")
        for f_train in listaFicheros_train:
            name = os.path.basename(f_train).split('.')[0]
            traintxt.write(name + "\n")
        for f_test in listaFicheros_test:
            name = os.path.basename(f_test).split('.')[0]
            testtxt.write(name + "\n")
            shutil.copy(f_test, os.path.join(outputPath, "JPEGImages"))

            ficherolabel = f_test[0:f_test.rfind('.')] + '.xml'
            ficherolabel = ficherolabel.replace("JPEGImages", "Annotations")  # obetenemos el nombre de los ficheros
            shutil.copy(ficherolabel, os.path.join(outputPath, "Annotations"))
            #     Aqui hemos usado la ruta para ir copiando los archivos de test en las carpetas correspondientes por que al estar ya la carpeta no podemos hacerlo de golpe
        # shutil.rmtree(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME))

    # def organize(self, train_percentage):
    #     super(MxNetDetector, self).organize( train_percentage)

    # def createModel(self):
    #     pass

    def train(self, framework_path = None):
        pass


    def evaluate(self, framework_path = None):
        pass

class VOCLike(gcv.data.VOCDetection):
    CLASSES = []

    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True, classes=[]):
        type(self).CLASSES=classes
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

def main():
    pass

if __name__ == "__main__":
    main()