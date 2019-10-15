#!/bin/sh

# Carga de librerías. Se trabaja con la versión 10 de cuda y 7 de cudnn
# que son con las que se ha compilado tensorflow. También es necesario
# cargar python.
module load cuda/10.1.105
module load cudnn/7.6
module load python/3.6.1
# Ejecutar fichero con configuración
#python3 kangaroo.py




