from objectDetectors.objectDetectionInterface import IObjectDetection
from imutils import paths
from Predictors.FSAFPredict import FSAFPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
import os
import shutil
from FSAF.train import trainModel

class FSAFDetector(IObjectDetection):
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
        shutil.copy(os.path.join(outputPath,"classes.csv"), os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_classes.csv"))

        """classescsv = open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, self.DATASET_NAME + "_classes.csv"), "w")
        classes = [cl for cl in open(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"))]
        rows = [c.replace('\n','')+","+ str(i)+"\n" for (i, c) in enumerate(classes) if i!=len(classes)-1]
        for row in rows:
            classescsv.write(row)
        classescsv.close()"""
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


    def train(self, framework_path = None, n_gpus = 1):
        batch_size = 4
        epochs = 25
        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+self.model)
        image_paths = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train","JPEGImages"), validExts=(".jpg")))
        n_steps = (len(image_paths)//batch_size)
        if (not (os.path.exists(os.path.join(outputPath,"models")))):
            os.makedirs(os.path.join(outputPath, "models"))


        class Aux():
            pass

        args = Aux()
        args.dataset_type='pascalCustom'
        args.pascal_path= outputPath
        args.imagenet_weights=True
        args.snapshot=None
        args.weights=None
        args.num_gpus=1
        args.lr=1e-4
        args.snapshot_path=os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models","fsaf_" + str(self.model) + '_' + self.DATASET_NAME)
        args.backbone = self.model
        args.gpu = 0
        args.random_transform=True
        args.compute_val_loss=False
        args.freeze_backbone=True
        args.batch_size = batch_size
        args.epochs=epochs
        args.steps=n_steps
        args.weighted_bifpn=False
        args.freeze_bn=False
        args.tensorboard_dir=False
        args.evaluation=False
        args.snapshots=True
        args.workers=1
        args.multiprocessing=False
        args.max_queue_size=10
        args.config=False
        args.image_min_side=800
        args.image_max_side=1333

        trainModel(args)
        
        shutil.rmtree(os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+self.model))

    def evaluate(self):
        fsafPredict = FSAFPredict(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models","fsaf_" + str(self.model) + '_' + self.DATASET_NAME,str(self.model)+'_pascalCustom_25.h5'),
            os.path.join(self.OUTPUT_PATH,self.DATASET_NAME, self.DATASET_NAME + "_classes.csv"),
            self.model)

        map = Map(fsafPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

def main():
    pass

if __name__ == "__main__":
    main()
