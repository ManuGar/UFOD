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
    frameworks = conf["frameworks"]
    exec = conf["exec"]
    type = exec["type"]
    params = exec["params"]
    n_gpus = exec["ngpus"]

    if (not (os.path.exists(os.path.join(".", "scripts")))):
        os.makedirs(os.path.join(".", "scripts"))
    if (not (os.path.exists(os.path.join(".", "datasets")))):
        os.makedirs(os.path.join(".", "datasets"))
    for fram, mod in frameworks:
        #aqui hay que crear un .sh para cada modelo que haya y meterle la llamada al train model con las variables necesarias
        file_name = os.path.join(".","scipts", "train_" + fram + "_" + mod + "_" + dataset_name + ".sh")
        f = open(file_name, "w")
        f.write("#!/bin/sh\n")
        if type =="slurm":
            if (fram == "Mxnet"):
                f.write("source configs_slurm/mxnet.sh\n")
            if (fram == "Rcnn"):
                f.write("source configs_slurm/maskrcnn.sh\n")
            if (fram == "Retinanet"):
                f.write("source configs_slurm/retinanet.sh\n")
            if (fram == "Tensorflow"):
                f.write("source configs_slurm/tensorflow.sh\n")
            if (fram == "Darknet"):
                f.write("source configs_slurm/yolo.sh\n")
        if type =="local":
            if (fram == "Mxnet"):
                f.write("source configs_local/mxnet.sh\n")
            if (fram == "Rcnn"):
                f.write("source configs_local/maskrcnn.sh\n")
            if (fram == "Retinanet"):
                f.write("source configs_local/retinanet.sh\n")
            if (fram == "Tensorflow"):
                f.write("source configs_local/tensorflow.sh\n")
            if (fram == "Darknet"):
                f.write("source configs_local/yolo.sh\n")
        f.write("python3 trainModel.py -f " + fram + " -m " + mod + " -d " + dataset + " -dn " + dataset_name)
        f.close()
        os.system("sbatch -p " + params["partition"] + " --gres=" + params["gres"] + " --time=" + params["time"] + " " + file_name + " --mem " + params["mem"])

if __name__ == "__main__":
    main()