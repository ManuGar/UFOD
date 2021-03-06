import os
import shutil
from imutils import paths
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree

class IObjectDetection(object):
    def __init__(self, dataset_path, dataset_name):
        self.DATASET =dataset_path
        self.DATASET_NAME =dataset_name
        self.OUTPUT_PATH = os.path.join(".","datasets")
        self.model = None
        if (not os.path.exists(self.OUTPUT_PATH)):
            os.makedirs(self.OUTPUT_PATH)
    # This function prepares the dataset to the structure of the selected framework
    def transform(self):
        pass
    # This function divides the dataset into train and test sets
    def split(self, train_percentage):
        if (not os.path.exists(os.path.join(self.DATASET, "test"))):
            datasetSplit(self.DATASET_NAME,self.OUTPUT_PATH, self.DATASET,train_percentage)
        else:
            copy_tree(self.DATASET, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME))

    def createModel(self):
        pass
    def train(self, framework_path = None, n_gpus = 1):
        pass
    def evaluate(self):
        pass


def datasetSplit(dataset_name, output_path, dataset_path, percentage):
    listaFicheros = list(paths.list_files(dataset_path, validExts=(".jpg")))
    train_list, test_list, _, _ = train_test_split(listaFicheros, listaFicheros, train_size=percentage, random_state=5)
    # creamos la estructura de carpetas, la primera contendra las imagenes del entrenamiento
    if (not os.path.exists(os.path.join(output_path, dataset_name,"train"))):
        os.makedirs(os.path.join(output_path, dataset_name, "train","JPEGImages"))
        os.makedirs(os.path.join(output_path, dataset_name, "train","Annotations"))
        os.makedirs(os.path.join(output_path, dataset_name, "test","JPEGImages"))
        os.makedirs(os.path.join(output_path, dataset_name, "test","Annotations"))
        os.makedirs(os.path.join(output_path, dataset_name, "models"))


    shutil.copy(os.path.join(dataset_path, "classes.names"), os.path.join(output_path, dataset_name, "classes.names"))
    for file in train_list:
        #Se obtiene el fichero de anotacion asociado
        ficherolabel = file[0:file.rfind('.')] + '.xml'
        ficherolabel = ficherolabel.replace("JPEGImages", "Annotations")
        # obetenemos el nombre de los ficheros
        name = str(os.path.basename(file).split('.')[0])
        image_splited = os.path.join(output_path, dataset_name, "train","JPEGImages", name + ".jpg")
        # movemos las imagenes a la carpeta JpegImages
        shutil.copy(file, image_splited)
        # movemos las anotaciones a la carpeta
        shutil.copy(ficherolabel, os.path.join(output_path, dataset_name, "train","Annotations",  name + ".xml"))
    # para las imagenes de entrenamiento
    for file in test_list:
        # obtenemos el fichero .txt asociado
        ficherolabel = file[0:file.rfind('.')] + '.xml'
        ficherolabel = ficherolabel.replace("JPEGImages", "Annotations")        # obetenemos el nombre de los ficheros
        name = str(os.path.basename(file).split('.')[0])
        image_splited = os.path.join(output_path, dataset_name, 'test',"JPEGImages", name + '.jpg')
        # movemos las imagenes a la carpeta JpegImages
        shutil.copy(file, image_splited)
        # movemos las anotaciones a la carpeta
        shutil.copy(ficherolabel, os.path.join(output_path, dataset_name, "test", "Annotations", name + '.xml'))