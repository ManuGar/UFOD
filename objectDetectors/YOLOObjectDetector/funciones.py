#! /usr/bin/python
# librerias
# import urllib2
try:
    import urllib.request as urllib
except ImportError:
    import urllib
from imutils import paths
import shutil
from sklearn.model_selection import train_test_split
import os


# from string import join
# funcion que nos va a generar el archivo fichero.data, donde tenemos las rutas de las imagenes
def generaFicheroData(darknetPath, NClases, Nproyecto):
    # creamos el fichero
    f = open(os.path.join(darknetPath, "cfg", Nproyecto + ".data"), 'w')
    # empezamos poniendo el numero de clases
    f.write('classes= ' + str(NClases) + '\n')
    f.write('train  = ' + Nproyecto + os.sep + "train.txt" + '\n')
    f.write('valid  = ' + Nproyecto + os.sep + "test.txt" + '\n')
    f.write('names = ' + "data" + os.sep + Nproyecto + ".names" + '\n')
    f.write('backup = backup' + '\n')
    f.close()


# generaFicheroData("cfg/fichero.data",20, "/home/pjreddie/data/voc/train.txt", "/home/pjreddie/data/voc/2007_test.txt", "data/voc.names", "/backup")

# funcion que nos va a generar el fichero names con .name
def generaFicheroNames(darknetPath, Nproyecto, clases):
    # creamos el fichero
    f = open(os.path.join(darknetPath, "data", Nproyecto + ".names"), 'w')
    # Separamos la lista de las clases por la coma
    trocitos_de_clases = clases.split(',')
    # recorremos la lista e imprimimos una clase por linea
    for element in trocitos_de_clases:
        f.write(element.strip() + '\n')


# generaFicheroNames("data/fichero.names", "aeroplane, bicycle,bird,boat,bottle,bus,car, cat, chair,cow,diningtable,dog, horse, motorbike, person, pottedplant, sheep, sofa,train,tvmonitor")
def contarClases(clases):
    numClases = 0
    trocitos_de_clases = clases.split(',')
    # recorremos la lista e imprimimos una clase por linea
    # for element in trocitos_de_clases:
    #     numClases = numClases + 1
    # return numClases

    return len(trocitos_de_clases)


# funcion que nos genera los ficheros de YOLO
def generaFicherosYoloTrain(darknetPath, Nproyecto, NClases):
    # creamos el fichero yolo.cfg con la configuracion correspondiente a YOLO
    f = open(os.path.join(darknetPath, "cfg", Nproyecto + "train.cfg"), 'w')
    # Texto del fichero
    mensaje = """[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((NClases + 5) * 3) + """
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=""" + str(NClases) + """
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((NClases + 5) * 3) + """
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=""" + str(NClases) + """
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((NClases + 5) * 3) + """
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=""" + str(NClases) + """
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1"""
    # escribimos el mensaje en el fichero
    f.write(mensaje)
    f.close()


# generaFicherosYoloTrain("prueba.cfg", 1)


def generaFicherosYoloTest(darknetPath, Nproyecto, NClases):
    # creamos el fichero yolo.cfg con la configuracion correspondiente a YOLO
    f = open(os.path.join(darknetPath, "cfg", Nproyecto + "test.cfg"), 'w')
    # Texto del fichero
    mensaje = """[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((NClases + 5) * 3) + """
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=""" + str(NClases) + """
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((NClases + 5) * 3) + """
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=""" + str(NClases) + """
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((NClases + 5) * 3) + """
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=""" + str(NClases) + """
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1"""
    # escribimos el mensaje en el fichero
    f.write(mensaje)
    f.close()


# generaFicherosYoloTest("prueba.cfg", 1)

# funcion que nos descarga los pesos para entrenar
def descargarPesos(url):
    file_name = url.split('/')[-1]
    u = urllib.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    # print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status) + 1)
        # print status,
    f.close()


# funcion que nos genera la instruccion para entrenar el modelo
def generaInstruccionEntrenar(Nproyecto, desdeCero):
    # hay dos formas de entrenar a YOLO, desde cero o usando unos pesos. Para eso tenemos el parametro desdeCero
    if desdeCero:
        # si es desde cero, no necesitamos pesos
        print('./darknet detector train cfg/' + Nproyecto + '.data cfg/' + Nproyecto + 'train.cfg')
    else:
        # si partimos de unos pesos preentrenados, sera necesario descargarlos
        descargarPesos('https://pjreddie.com/media/files/darknet53.conv.74')
        print('./darknet detector train cfg/' + Nproyecto + '.data cfg/' + Nproyecto + 'train.cfg darknet53.conv.74')


# generaInstruccionEntrenar("cfg/vocEstomas.data","cfg/yolov3.cfg", False)

# funcion que nos genera la instruccion para evaluar el modelo
def generaInstruccionEvaluar(Nproyecto, pathPesos, threshold=0.25):
    if (os.path.exists(pathPesos)):
        print(
            './darknet detector map cfg/' + Nproyecto + '.data cfg/' + Nproyecto + 'test.cfg ' + pathPesos + ' -thresh ' + str(
                threshold))
    else:
        print('no existen esos pesos')


# generaInstruccionEvaluar("cfg/vocEstomas.data","cfg/yolov3.cfg","backup/yolov3_220000.weights", 0.5)

# funcion que nos genera la instruccion para predecir dada una foto
def generaInstruccionPredecir(Nproyecto, pathPesos, pathFoto, threshold=0.25):
    if (os.path.exists(pathPesos)):
        if (os.path.exists(pathFoto)):
            print(
                './darknet detector test cfg/' + Nproyecto + '.data cfg/' + Nproyecto + 'test.cfg ' + pathPesos + ' ' + pathFoto + ' -thresh ' + str(
                    threshold))
        else:
            print('no existe la imagen')
    else:
        print('no existen esos pesos')


# generaInstruccionPredecir("cfg/vocEstomas.data","cfg/yolov3.cfg","backup/yolov3_220000.weights","estomas/test/JPEGImages/0_0_1044.1.B3.jpg",0.5)

# funcion que comprueba si dada una carpeta hay imagenes .jpg
def compruebeImages(pathImages):
    listaFicheros = list(paths.list_files(pathImages, validExts=(".jpg")))
    if len(listaFicheros) == 0:
        print('En esta carpeta no hay imagenes en formato .jpg')
    else:
        print('Todo correcto')


# funcion que dada una ruta comprueba si existen los ficheros .txt que poseen las anotaciones
def compruebeTXT(pathImages):
    listaFicheros = list(paths.list_files(pathImages, validExts=(".jpg")))
    for file in listaFicheros:
        ficheroAnotaciones = file.split(".jpg")[0] + ".txt"
        if (os.path.exists(ficheroAnotaciones)):
            print('Todo correcto, continuamos')
        else:
            print('No existe el fichero de anotaciones')


def datasetSplit(Nproyecto, darknetPath, pathImages, porcentaje):
    listaFicheros = list(paths.list_files(pathImages, validExts=(".jpg")))
    train_list, test_list, _, _ = train_test_split(listaFicheros, listaFicheros, train_size=porcentaje)
    # creamos la estructura de carpetas, la primera contendra las imagenes del entrenamiento
    os.makedirs(os.path.join(darknetPath, Nproyecto, 'train', 'JPEGImages'))
    # esta carpeta contendra las anotaciones de las imagenes de entrenamiento
    os.makedirs(os.path.join(darknetPath, Nproyecto, 'train', 'labels'))
    # y esta ultima carpeta va a contener tanto las imagenes como los ficheros de anotaciones del test
    os.makedirs(os.path.join(darknetPath, Nproyecto, 'test', 'JPEGImages'))
    # para las imagenes de entrenamiento
    for file in train_list:
        # obtenemos el fichero .txt asociado
        ficherolabel = file[0:file.rfind('.')] + '.txt'
        # obetenemos el nombre de los ficheros
        name = os.path.basename(file).split('.')[0]
        # movemos las imagenes a la carpeta JpegImages
        shutil.copy(file, os.path.join(darknetPath, Nproyecto, 'train', 'JPEGImages', name + '.jpg'))
        # movemos las anotaciones a la carpeta
        shutil.copy(ficherolabel, os.path.join(darknetPath, Nproyecto, 'train', 'labels', name + '.txt'))
    # para las imagenes de entrenamiento
    for file in test_list:
        # obtenemos el fichero .txt asociado
        ficherolabel = file[0:file.rfind('.')] + '.txt'
        # obetenemos el nombre de los ficheros
        name = os.path.basename(file).split('.')[0]
        # movemos las imagenes a la carpeta JpegImages
        shutil.copy(file, os.path.join(darknetPath, Nproyecto, 'test', 'JPEGImages', name + '.jpg'))
        # movemos las anotaciones a la carpeta
        shutil.copy(ficherolabel, os.path.join(darknetPath, Nproyecto, 'test', 'JPEGImages', name + '.txt'))


def generaFicheroTrain(darknetPath, Nproyecto):
    # creamos el fichero train.txt
    f = open(os.path.join(darknetPath, Nproyecto, "train.txt"), 'w')
    # listamos todos los ficheros .jpg del conjunto de entrenamiento
    files = os.listdir(os.path.join(darknetPath, Nproyecto, "train/JPEGImages/"))
    # recorremos la lista e imprimimos una imagen por linea
    for l in files:
        f.write(os.path.join(darknetPath, Nproyecto, "train/JPEGImages/", l) + '\n')


def generaFicheroTest(darknetPath, Nproyecto):
    # creamos el fichero test.txt
    f = open(os.path.join(darknetPath, Nproyecto, "test.txt"), 'w')
    # listamos todos los ficheros .jpg del conjunto de test
    files = os.listdir(os.path.join(darknetPath, Nproyecto, "test/JPEGImages/"))
    # recorremos la lista e imprimimos una imagen por linea
    for l in files:
        start, ext = os.path.splitext(l)
        if ext == ('.jpg'):
            # si es una imagen la guardamos
            f.write(os.path.join(darknetPath, Nproyecto, "test/JPEGImages/", l) + '\n')


# funcion que nos va a listar todos los pesos creados
def listarBackup(darknetPath):
    contenido = os.listdir(os.path.join(darknetPath, 'backup/'))
    for elemento in contenido:
        print(elemento)

