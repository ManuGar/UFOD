from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.MXNetObjectDetector import  functions as fn

# from gluoncv import model_zoo, data, utils
# from mxnet import gluon, autograd
# from mxnet.gluon.model_zoo import vision as models
# from mxnet.gluon import data, utils
# from mxnet.gluon.data.vision import datasets, transforms
import gluoncv as gcv
# import mxnet as mx
import shutil
import os



class MxNetDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name):
        IObjectDetection.__init__(self, dataset_path, dataset_name)

    def transform(self):
        pass

    def organize(self, train_percentage):
        # dataset_name, darknet_path, dataset_path, train%_split
        # this function prepare the dataset to the yolo estructure
        # dataset_name = self.DATASET[self.DATASET.rfind(os.sep) + 1:]
        fn.datasetSplit(self.DATASET_NAME, self.OUTPUT_PATH, self.DATASET, train_percentage)
        shutil.copy(os.path.join(self.DATASET, "classes.names"),
                    os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME))

    def createModel(self):
        pass

    def train(self, framework_path = None):
        pass


    def evaluate(self, framework_path = None):
        pass

class VOCLike(gcv.data.VOCDetection):
    CLASSES = []

    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

def main():
    pass

if __name__ == "__main__":
    main()