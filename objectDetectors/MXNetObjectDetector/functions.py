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

    # # creamos la estructura de carpetas, la primera contendra las imagenes del entrenamiento
    # os.makedirs(os.path.join(outputPath, Nproyecto, 'train', 'JPEGImages'))
    # # esta carpeta contendra las anotaciones de las imagenes de entrenamiento
    # os.makedirs(os.path.join(outputPath, Nproyecto, 'train', 'labels'))
    # # y esta ultima carpeta va a contener tanto las imagenes como los ficheros de anotaciones del test
    # os.makedirs(os.path.join(outputPath, Nproyecto, 'test', 'JPEGImages'))
    # # para las imagenes de entrenamiento

    traintxt = open(os.path.join(outputPath, "ImageSets","Main", "train.txt"), "w")
    testtxt = open(os.path.join(outputPath, "ImageSets","Main", "test.txt"), "w")
    for file in train_list:
        # obtenemos el fichero .txt asociado
        # ficherolabel = file[0:file.rfind('.')] + '.txt'
        # obetenemos el nombre de los ficheros
        name = os.path.basename(file).split('.')[0]
        # image_splited = os.path.join(outputPath, 'JPEGImages', name)
        traintxt.write(name + "\n")
        # traintxt.write(os.path.abspath(image_splited) + "\n")
        # movemos las imagenes a la carpeta JpegImages
        # shutil.copy(file, image_splited)
        # movemos las anotaciones a la carpeta
        # shutil.copy(ficherolabel, os.path.join(outputPath, Nproyecto, 'train', 'labels', name + '.xml'))
    # para las imagenes de entrenamiento
    for file in test_list:
        # obtenemos el fichero .txt asociado
        # ficherolabel = file[0:file.rfind('.')] + '.txt'
        # obetenemos el nombre de los ficheros
        name = os.path.basename(file).split('.')[0]
        # image_splited = os.path.join(outputPath,'JPEGImages', name + '.jpg')
        # testtxt.write(os.path.abspath(image_splited) + "\n")
        testtxt.write(name+ "\n")
        # movemos las imagenes a la carpeta JpegImages
        # shutil.copy(file, image_splited)
        # movemos las anotaciones a la carpeta
        # shutil.copy(ficherolabel, os.path.join(outputPath, Nproyecto, 'test', 'JPEGImages', name + '.xml'))

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
