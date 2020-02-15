from objectDetectors.objectDetectionInterface import IObjectDetection
from imutils import paths
from Predictors.EfficientdetPredict import EfficientdetPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
import os
import shutil
from EfficientDet.train import trainModel

class EfficientDetDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name,model):
        IObjectDetection.__init__(self, dataset_path, dataset_name)
        self.model = model

    def transform(self):
        listaFicheros_train = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train"), validExts=(".jpg")))
        listaFicheros_test = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"test"), validExts=(".jpg")))

        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+str(self.model))
        # outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME)

        shutil.copytree(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","JPEGImages"), os.path.join(outputPath, "JPEGImages"))
        shutil.copytree(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","Annotations"), os.path.join(outputPath, "Annotations"))
        if (not (os.path.exists(os.path.join(outputPath, "ImageSets")))):
            os.makedirs(os.path.join(outputPath, "ImageSets", "Main"))

        shutil.copy(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"), outputPath)

        classescsv = open(os.path.join(outputPath,"classes.csv"), "w")
        with open(os.path.join(outputPath,"classes.names")) as f:
            classes = f.read()
            classes = classes.split('\n')
        rows = [",".join([c, str(i)]) for (i, c) in enumerate(classes)]
        classescsv.write("\n".join(rows))
        classescsv.close()

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
        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+str(self.model))
        image_paths = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","JPEGImages"), validExts=(".jpg")))

        n_steps = (len(image_paths)//batch_size)


        if (not (os.path.exists(os.path.join(outputPath,"models")))):
            os.makedirs(os.path.join(outputPath, "models"))

        class Aux():
            pass

        args = Aux()
        args.dataset_type='pascalCustom'
        args.pascal_path= outputPath
        args.snapshot='imagenet'
        args.snapshot_path=os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models","efficientdet" + str(self.model) + '_' + self.DATASET_NAME)
        args.phi = 0
        args.gpu = 0
        args.random_transform=True
        args.compute_val_loss=True
        args.freeze_backbone=True
        args.batch_size = batch_size
        args.epochs=1
        args.steps=n_steps
        args.weighted_bifpn=False
        args.freeze_bn=False
        args.tensorboard_dir=False
        args.evaluation=False
        args.snapshots=True
        args.workers=1
        args.multiprocessing=False
        args.max_queue_size=10

        trainModel(args)


        args.freeze_bn=True
        args.freeze_backbone=False
        args.epochs=1
        args.snapshot=os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models","efficientdet" + str(self.model) + '_' + self.DATASET_NAME,'pascalCustom_'+str(args.epochs)+'.h5')
        trainModel(args)

        shutil.rmtree(os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+self.model))



    def evaluate(self):
        efficientdetPredict = EfficientdetPredict(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models","efficientdet" + str(self.model) + '_' + self.DATASET_NAME,'pascalCustom_01.h5'),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME + "_classes.csv"),
            self.model)

        map = Map(efficientdetPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

def main():
    pass

if __name__ == "__main__":
    main()
