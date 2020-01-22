import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

# dataset_path(la del map)
def visualize(resulst_path):
    results = [p for p in os.listdir(resulst_path) if  p.endswith(".txt") and "results" in p]
    classes = []
    models = []
    resultsClasses = []
    measures = ["precision", "recall", "f1-score", "mAP"]
    resultsModels = []
    for r in results:
        models.append(r.replace("results.txt",""))
        r = open(os.path.join(resulst_path,r))
        for line in r:
            if "class_id" in line:
                line = line.split(",")
                name = line[1]
                name = name.split(" = ")[-1]
                if(name not in classes):
                    classes.append(name)
                measure = line[-1]
                measure = measure.split()[-2]
                resultsClasses.append(float(measure))

            if "F1-score" in line:
                measuresModel = line.split(", ")
                precision =100* float(measuresModel[0].split(" = " )[1])
                recall =100* float(measuresModel[1].split(" = ")[1])
                f1 =100* float(measuresModel[2].split(" = ")[1])
                resultsModels.append(precision)
                resultsModels.append(recall)
                resultsModels.append(f1)
            if "(mAP)" in line:
                map = float(line.split()[-2])
                resultsModels.append(map)


    resultsClasses = np.array(resultsClasses)
    shape = (len(models), len(classes))
    resultsClasses=resultsClasses.reshape(shape)
    resultsClasses=resultsClasses.transpose()
    resultsClasses=resultsClasses.flatten()


    # This code generates the plot for the performance of the models for each class

    y = np.array([np.repeat(i + 1, len(models)) for i, _ in enumerate(classes)]).flatten()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
    colors = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'cyan', 'yellow', 'brown', 'black']
    colors = colors[:len(models)]
    col = []
    for _ in classes:
        col.append(colors)
    col = np.array(col).flatten()
    axes.yaxis.grid(True)
    axes.set_xlabel("AP")
    axes.xaxis.grid(True)
    axes.xaxis.set_major_locator(plt.MaxNLocator(5))
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.scatter(resultsClasses, y, c=col, s=200, label=models)
    plt.yticks(range(1, len(classes) + 1), classes)
    plt.xlim(0, 100)
    recs = []
    for i in range(0, len(colors)):
        recs.append(mpatches.Circle((0, 0), radius=1, fc=colors[i]))
    plt.legend(recs, models, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.savefig("classesPerfomance.png")




    resultsModels = np.array(resultsModels)
    shape = (len(models), len(measures))
    resultsModels = resultsModels.reshape(shape)
    resultsModels = resultsModels.transpose()
    resultsModels = resultsModels.flatten()

    # This code generates the plot for the performance of each model with different measures
    y = np.array([np.repeat(i + 1, len(models)) for i, _ in enumerate(measures)]).flatten()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
    colors = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'cyan', 'yellow', 'brown', 'black']
    colors = colors[:len(models)]
    col = []
    for _ in measures:
        col.append(colors)
    col = np.array(col).flatten()
    axes.yaxis.grid(True)
    axes.set_xlabel("%")
    axes.xaxis.grid(True)
    axes.xaxis.set_major_locator(plt.MaxNLocator(5))
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.scatter(resultsModels, y, c=col, s=200, label=models)
    plt.yticks(range(1, len(measures) + 1), measures)
    plt.xlim(0, 100)
    recs = []
    for i in range(0, len(colors)):
        recs.append(mpatches.Circle((0, 0), radius=1, fc=colors[i]))
    plt.legend(recs, models, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.savefig(os.path.join(resulst_path,"modelsPerfomance.png"))

