import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 30})

# dataset_path(la del map)
def visualize(dataset_path):
    results = [p for p in os.listdir(dataset_path) if  p.endswith(".data") and "results" in p]
    classes = []
    models = []
    resultsClasses = []
    resultsModels = []
    for r in results:
        models.append(r.replace("results",""))
        r = open(r)
        for line in r:
            print(line)
            if "class_id" in line:
                line = line.split(",")
                name = line[1]
                name = name.split("=")[0]
                classes.append(name)
                print(name)
                print("ESTE ES EL NOMBRE DE LA CLASE")
                measure = line[2]
                measure = measure.split("=")[0].split("%")[0]
                resultsClasses.append(measure)
                print(measure)
                print("ESTE ES EL RESULTADO DE LA CLASE")
            if "mAP" in line:
                measureModel = line.split()[-2]
                resultsModels.append(measureModel)
                print(measureModel)
                print("ESTE ES EL RESUTADO DE LO QUE HA SACADO EL MAP DEL MODELO")

