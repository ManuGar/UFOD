import argparse
import factoryModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=True, help="framework of the object detection model")
    ap.add_argument("-m", "--model", required=True, help="deep learning object detection model")
    ap.add_argument("-d", "--dataset", required=True, help="path of the dataset")
    ap.add_argument("-dn", "--dataset_name", required=True, help="name of the dataset")
    ap.add_argument("-o", "--output_path", required=True, help="path of the output of the framework")


    args = vars(ap.parse_args())
    framework = args["framework"]
    modelText = args["model"]
    dataset = args["dataset"]
    dataset_name = args["dataset_name"]
    output_path = args["output_path"]
    model = factoryModel.createModel(framework, modelText, dataset, dataset_name, output_path)
    model.transform()
    model.organize()
    model.createModel()
    model.train()

if __name__ == "__main__":
    main()