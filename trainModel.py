import argparse
import factoryModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=True, help="framework of the object detection model")
    ap.add_argument("-m", "--model", required=True, help="deep learning object detection model")
    ap.add_argument("-d", "--dataset", required=True, help="path of the dataset")
    args = vars(ap.parse_args())
    modelText = args["model"]
    dataset = args["dataset"]
    model = factoryModel.createModel(modelText)
    model.transform(dataset,output_path)
    model.organize(dataset)
    model.createModel(dataset)
    model.train(dataset)

if __name__ == "__main__":
    main()