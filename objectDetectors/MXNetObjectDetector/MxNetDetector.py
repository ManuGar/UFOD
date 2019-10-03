from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.MXNetObjectDetector import  functions as fn




# Creo que el model zoo de esta libreria y la de mxnet son las mismas o muy parecidas asi que intentar usar solo la de mxnet por depencias
from gluoncv import model_zoo, data, utils
from mxnet import gluon, autograd
# from mxnet.gluon.model_zoo import vision as models
# from mxnet.gluon import data, utils
from mxnet.gluon.data.vision import datasets, transforms
import gluoncv as gcv
import mxnet as mx
import matplotlib.pyplot as plt
import shutil
import os
import time



class MxNetDetector(IObjectDetection):
    def __init__(self):
        IObjectDetection.__init__(self)

    def transform(self, dataset_path, output_path):
        pass

    def organize(self, dataset_path, output_path, train_percentage):
        # dataset_name, darknet_path, dataset_path, train%_split
        # this function prepare the dataset to the yolo estructure
        dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
        fn.datasetSplit(dataset_name, output_path, dataset_path, train_percentage)
        shutil.copy(os.path.join(dataset_path, "classes.names"),
                    os.path.join(output_path, "VOC" + dataset_name))

    def createModel(self, dataset_path):
        pass

    def train(self, dataset_path, dataset_name):
        pass


    def evaluate(self, dataset_path):
        pass

class VOCLike(gcv.data.VOCDetection):
    CLASSES = []

    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

def main():
    pass

if __name__ == "__main__":
    main()