from objectDetectors.objectDetectionInterface import IObjectDetection
from imutils import paths
from Predictors.FcosPredict import FcosPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
import os
import shutil

class FcosDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name,model):
        IObjectDetection.__init__(self, dataset_path, dataset_name)
        self.model = model

    def transform(self):
        listaFicheros_train = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train"), validExts=(".jpg")))
        listaFicheros_test = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"test"), validExts=(".jpg")))

        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+self.model)
        # outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME)

        shutil.copytree(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","JPEGImages"), os.path.join(outputPath, "JPEGImages"))
        shutil.copytree(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","Annotations"), os.path.join(outputPath, "Annotations"))
        if (not (os.path.exists(os.path.join(outputPath, "ImageSets")))):
            os.makedirs(os.path.join(outputPath, "ImageSets", "Main"))

        shutil.copy(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"), outputPath)
        traintxt = open(os.path.join(outputPath, "ImageSets", "Main", "train.txt"), "w")
        testtxt = open(os.path.join(outputPath, "ImageSets", "Main", "test.txt"), "w")

        classescsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_classes.csv"), "w")
        classes = [cl for cl in open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"))]
        rows = [",".join([c, str(i)]) for (i, c) in enumerate(classes)]
        classescsv.write("\n".join(rows))
        classescsv.close()
        for f_train in listaFicheros_train:
            name = os.path.basename(f_train).split('.')[0]
            traintxt.write(name + "\n")
        for f_test in listaFicheros_test:
            name = os.path.basename(f_test).split('.')[0]
            testtxt.write(name + "\n")
            shutil.copy(f_test, os.path.join(outputPath, "JPEGImages"))

            ficherolabel = f_test[0:f_test.rfind('.')] + '.xml'
            ficherolabel = ficherolabel.replace("JPEGImages", "Annotations")  # obetenemos el nombre de los ficheros
            shutil.copy(ficherolabel, os.path.join(outputPath, "Annotations"))
        # shutil.rmtree(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME))


    def train(self, framework_path = None, n_gpus = 1):
        batch_size = 4
        epochs = 25
        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+self.model)
        image_paths = list(paths.list_files(os.path.join(outputPath,self.DATASET_NAME,"train"), validExts=(".jpg")))
        n_steps = (len(image_paths)/batch_size)
        if (not (os.path.exists(os.path.join(outputPath,"models")))):
            os.makedirs(os.path.join(outputPath, "models"))
        os.system("python3 "+ framework_path + "/train.py --backbone " + self.model + " --snapshot-path " + os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models","fcos_" + self.model + '_' + self.DATASET_NAME + '.h5') + " --gpu 0 --batch-size " + batch_size + " --epochs " + epochs + " --steps " + n_steps + " pascalCustom " + outputPath)

        shutil.rmtree(os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+self.model))

    def evaluate(self):
        fcosPredict = FcosPredict(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME,"models","fcos_" + self.model + '_' + self.DATASET_NAME + '.h5'),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME + "_classes.csv"),
            self.model)

        map = Map(fcosPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

def main():
    pass

if __name__ == "__main__":
    main()