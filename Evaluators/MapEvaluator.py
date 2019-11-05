from Evaluators.IEvaluator import IEvaluator
from imutils import paths
import os
import shutil

class MapEvaluator(IEvaluator):
    def __init__(self, predictor):
        self.predictor = predictor
    def evaluate(self, dataset_name, dataset_path):
        aux_path = os.path.join("..", "map", dataset_name)
        if (not (os.path.exists(aux_path))):
            os.makedirs(aux_path)
        os.makedirs(os.path.join(aux_path,"labels"))
        image_paths = list(paths.list_files(os.path.join(dataset_path, "test"), validExts=(".jpg")))
        for im in image_paths:
            shutil.copy(im, os.path.join(aux_path,"labels"))




def main():
    pass

if __name__ == "__main__":
    main()
