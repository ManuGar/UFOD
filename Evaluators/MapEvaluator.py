from Evaluators.IEvaluator import IEvaluator
from imutils import paths
import os
import shutil

class MapEvaluator(IEvaluator):
    def __init__(self, predictor, dataset_name,dataset_path):
        super().__init__(predictor,dataset_name,dataset_path)
    def evaluate(self):
        aux_path = os.path.join("..", "map", self.dataset_name)
        if (not (os.path.exists(aux_path))):
            os.makedirs(os.path.join(aux_path,"labels"))
        shutil.copy(os.path.join(self.dataset_path, "classes.names"), os.path.join(aux_path, "classes.names"))

        image_paths = list(paths.list_files(os.path.join(self.dataset_path, "test"), validExts=(".jpg")))
        for image in image_paths:
            shutil.copy(image, os.path.join(aux_path,"labels"))
            shutil.copy(image, os.path.join(aux_path,"detection"))
            name = os.path.basename(image).split('.')[0]
            anno_splited = os.path.join(self.dataset_path, "test", "Annotations", name + ".xml")
            shutil.copy(anno_splited, os.path.join(aux_path,"labels"))
        os.system("python3 map/pascal2yolo_labels.py -d " + os.path.join(os.path.join(aux_path,"labels") + " -f " + os.path.join(aux_path,"classes.names")))
        self.predictor.predict(os.path.join(aux_path,"detection"))
        # In this moment we have the images predicted. Now we are going to tranform the predicted annotations to Darknet format
        os.system("python3 map/pascal2yolo_detection.py -d " + os.path.join(
            os.path.join(aux_path,"detection") + " -f " + os.path.join(aux_path, "classes.names")))
        os.system("find `pwd`map/" + self.dataset_name+"/labels -name '*.txt' > " + aux_path + "/test.txt")
        os.system("map/darknet detector compare " + os.path.join(aux_path,"test.txt") + " " + os.path.join(aux_path,"classes.names") + " > " + aux_path + "/results.txt")

def main():
    pass

if __name__ == "__main__":
    main()
