import argparse
import factoryModel
from conf import Conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=True, help="framework of the object detection model")
    ap.add_argument("-m", "--model", required=True, help="deep learning object detection model")
    ap.add_argument("-d", "--dataset", required=True, help="path of the dataset")
    ap.add_argument("-dn", "--dataset_name", required=True, help="name of the dataset")
    ap.add_argument("-ng", "--ngpus", required=False, help="number of gpus to use")

    args = vars(ap.parse_args())

    framework = args["framework"]
    modelText = args["model"]
    dataset = args["dataset"]
    dataset_name = args["dataset_name"]
    ngpus = args["ngpus"]


    conf = Conf("./config_framework.json")

    model = factoryModel.createModel(framework, modelText, dataset, dataset_name)
    model.organize(0.75)
    model.transform()
    model.createModel()
    model.train(conf[framework], ngpus) #En algunos casos tendremos que aniadir el path del framework para que pueda trabajar con el

if __name__ == "__main__":
    main()