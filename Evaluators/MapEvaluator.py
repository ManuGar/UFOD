from Evaluators.IEvaluator import IEvaluator
from imutils import paths
import os
import shutil

class MapEvaluator(IEvaluator):
    def __init__(self, predictor, dataset_name,dataset_path, model_name):
        super().__init__(predictor,dataset_name,dataset_path, model_name)
    def evaluate(self):
        aux_path = os.path.join("map", self.dataset_name)
        if (not (os.path.exists(aux_path))):
            os.makedirs(os.path.join(aux_path))
        if( not (os.path.exists(os.path.join(aux_path,"labels")))):
            os.makedirs(os.path.join(aux_path,"labels"))
        if (not (os.path.exists(os.path.join(aux_path, "detection")))):
            os.makedirs(os.path.join(aux_path,"detection"))

        shutil.copy(os.path.join(self.dataset_path, "classes.names"), os.path.join(aux_path, "classes.names"))

        image_paths = list(paths.list_files(os.path.join(self.dataset_path, "test"), validExts=(".jpg")))
        for image in image_paths:
            name = os.path.basename(image).split('.')[0]
            shutil.copy(image, os.path.join(aux_path,"labels",os.path.basename(image)))
            shutil.copy(image, os.path.join(aux_path,"detection",os.path.basename(image)))
            anno_splited = os.path.join(self.dataset_path, "test", "Annotations", name + ".xml")
            shutil.copy(anno_splited, os.path.join(aux_path,"labels",name + ".xml"))
        os.system("python3 map/pascal2yolo_labels.py -d " + os.path.join(os.path.join(aux_path,"labels/") + " -f " + os.path.join(aux_path,"classes.names")))
        self.predictor.predict(os.path.join(aux_path,"detection/"))
        # In this moment we have the images predicted. Now we are going to tranform the predicted annotations to Darknet format
        os.system("python3 map/pascal2yolo_detection.py -d " + os.path.join(
            os.path.join(aux_path,"detection/") + " -f " + os.path.join(aux_path, "classes.names")))
        os.system("find `pwd`/map/" + self.dataset_name+"/labels -name '*.txt' > " + aux_path + "/test.txt")
        os.system("./map/darknet detector compare " + os.path.join(aux_path,"test.txt") + " " + os.path.join(aux_path,"classes.names") + " > " + aux_path + "/" + self.model_name+ "results.txt")
        # shutil.rmtree(os.path.join(aux_path,"detection"))
        shutil.move(os.path.join(aux_path,"detection"),os.path.join(aux_path,"detection"+self.model_name))
        shutil.rmtree(os.path.join(aux_path,"labels"))

        # Esto es por si se quiere mover el archivo con lo resultados para que esten todos en la misma ubicacion
        shutil.copy( aux_path + "/" + self.model_name+ "results.txt", os.path.join( "..", os.sep, "datasets",self.dataset_name))
        os.remove(os.path.join(aux_path,"classes.names"))


def main():
    pass

if __name__ == "__main__":
    main()
