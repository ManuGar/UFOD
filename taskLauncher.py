import argparse
from conf import Conf
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path of the configuration file")
    args = vars(ap.parse_args())

    config = args["conf"]

    conf = Conf(config)
    dataset = conf["dataset"]
    dataset_name = conf["dataset_name"]
    exec_time = conf["exec_time"]
    frameworks = conf["frameworks"]

    for fram, mod in frameworks:
        #aqui hay que crear un .sh para cada modelo que haya y meterle la llamada al train model con las variables necesarias
        file_name = "train_" + fram + "_" + mod + "_" + dataset_name + ".sh"
        f = open(file_name, "w")
        f.write("#!/bin/sh\n")

        if (fram == "Mxnet"):
            f.write("source configs/mxnet.sh\n")
        if (fram == "Rcnn"):
            f.write("source configs/maskrcnn.sh\n")
        if (fram == "Retinanet"):
            f.write("source configs/retinanet.sh\n")
        if (fram == "Tensorflow"):
            f.write("source configs/tensorflow.sh\n")
        if (fram == "Darknet"):
            f.write("source configs/yolo.sh\n")
        f.write("python trainModel.py -f " + fram + " -m " + mod + " -d " + dataset + " -dn " + dataset_name)
        f.close()
        os.system("sbatch -p gpu --gres=gpu:kepler:2 --time=" + exec_time + " " + file_name)

if __name__ == "__main__":
    main()