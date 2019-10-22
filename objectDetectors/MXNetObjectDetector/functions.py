try:
    import urllib.request as urllib
except ImportError:
    import urllib
from imutils import paths
from sklearn.model_selection import train_test_split
from mxnet import autograd,gluon
import mxnet as mx
import shutil
import os


def datasetSplit(Nproyecto, outputPath, pathImages, porcentaje):
    listaFicheros = list(paths.list_files(pathImages, validExts=(".jpg")))
    train_list, test_list, _, _ = train_test_split(listaFicheros, listaFicheros, train_size=porcentaje, random_state=5)
    outputPath = os.path.join(outputPath, "VOC" + Nproyecto)
    shutil.copytree(os.path.join(pathImages,"Annotations") , os.path.join(outputPath, "Annotations"))
    shutil.copytree(os.path.join(pathImages,"JPEGImages") , os.path.join(outputPath, "JPEGImages"))
    if (not (os.path.exists(os.path.join(outputPath, Nproyecto, "ImageSets")))):
        os.makedirs(os.path.join(outputPath, "ImageSets","Main"))

    traintxt = open(os.path.join(outputPath, "ImageSets","Main", "train.txt"), "w")
    testtxt = open(os.path.join(outputPath, "ImageSets","Main", "test.txt"), "w")
    for file in train_list:
        name = os.path.basename(file).split('.')[0]
        traintxt.write(name + "\n")

    for file in test_list:
        name = os.path.basename(file).split('.')[0]
        testtxt.write(name+ "\n")

def readClasses(dataset_path):
    classes = []
    fil = open(os.path.join(dataset_path, "classes.names"))
    for cl in fil:
        cl = cl.split("\n")[0]
        classes.append(cl)
    return classes

def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader
