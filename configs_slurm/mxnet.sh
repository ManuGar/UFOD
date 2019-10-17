#!/bin/sh

# Carga de librerías. Se trabaja con la versión 10 de cuda y 7 de cudnn
# También es necesario cargar python.
module load cuda/10.1.105
module load cudnn/7.6
module load python/3.6.1
# Cargar el entorno virtual
source ~/mxnet/bin/activate
# Ejecutar fichero con código a ejecutar
#python3 finetune_detection.py -c kangaroo_config.py
#exit 0
