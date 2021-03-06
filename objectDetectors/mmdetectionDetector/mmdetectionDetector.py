from objectDetectors.objectDetectionInterface import IObjectDetection
from imutils import paths
from Predictors.mmdetectionPredict import mmdetectionPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
import os
import shutil
import sys
import glob
import xml.etree.ElementTree as ET

sys.path.append("mmdetection")


class mmdetectionDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name,model):
        IObjectDetection.__init__(self, dataset_path, dataset_name)
        self.model = model

    def transform(self):
        MODELS_CONFIG = {'faster_rcnn_r50_fpn_1x': {'config_file': 'configs/faster_rcnn_r50_fpn_1x.py'},
                         'faster_rcnn_r101_fpn_1x': {'config_file': 'configs/faster_rcnn_r101_fpn_1x'},
                         'cascade_rcnn_r50_fpn_1x': {'config_file': 'configs/cascade_rcnn_r50_fpn_1x.py'},
                         'cascade_rcnn_r101_fpn_1x': {'config_file': 'configs/cascade_rcnn_r101_fpn_1x.py'},
                         'cascade_rcnn_x101_32x4d_fpn_1x': {'config_file': 'configs/cascade_rcnn_x101_32x4d_fpn_1x.py'},
                         'retinanet_r50_fpn_1x': {'config_file': 'configs/retinanet_r50_fpn_1x.py'},
                         'retinanet_r101_fpn_1x': {'config_file': 'configs/retinanet_r101_fpn_1x.py'},
                        'rpn_r50_fpn_1x': {'config_file': 'configs/rpn_r50_fpn_1x'},
                        'rpn_r101_fpn_1x': {'config_file': 'configs/rpn_r101_fpn_1x'}}

        selected_model = self.model
        config_file = MODELS_CONFIG[selected_model]['config_file']
        total_epochs = 20
        listaFicheros_train = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train"), validExts=(".jpg")))
        listaFicheros_test = list(paths.list_files(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"test"), validExts=(".jpg")))

        outputPath = os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+str(self.model),"VOC2007")
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

        traintxt = open(os.path.join(outputPath, "ImageSets", "Main", "trainval.txt"), "w")
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

        
        anno_path = os.path.join(outputPath, "Annotations")
        voc_file = os.path.join("mmdetection", "mmdet/datasets/voc.py")
        classes_names = []
        xml_list = []
        for xml_file in glob.glob(anno_path + "/*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall("object"):
                classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        classes_names.sort()
        import re

        fname = voc_file
        with open(fname) as f:
            s = f.read()
            s = re.sub('CLASSES = \(.*?\)','CLASSES = ({})'.format(", ".join(["\'{}\'".format(name) for name in classes_names])), s, flags=re.S)

        with open(fname, 'w') as f:
            f.write(s)  
        
        config_fname = os.path.join('mmdetection', config_file)
        shutil.copy(config_fname, config_fname[0:config_fname.rfind(".")] +self.DATASET_NAME   +".py")
        self.config_fname = config_fname[0:config_fname.rfind(".")] +self.DATASET_NAME   +".py"
        #### Modificando el fichero de configuración
        fname = self.config_fname
        with open(fname) as f:
            s = f.read()
            work_dir = re.findall(r"work_dir = \'(.*?)\'", s)[0]
            # Update `num_classes` including `background` class.
            s = re.sub('num_classes=.*?,',
                       'num_classes={},'.format(len(classes_names) + 1), s)
            s = re.sub('ann_file=.*?\],',
                       "ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',", s, flags=re.S)
            s = re.sub('total_epochs = \d+',
                       'total_epochs = {} #'.format(total_epochs), s)
            if "CocoDataset" in s:
                s = re.sub("dataset_type = 'CocoDataset'",
                           "dataset_type = 'VOCDataset'", s)
                
                s = re.sub("data_root = 'data/coco/'",
                           "data_root = \'"+ outputPath[0:outputPath.rfind("/")] +"/\'", s)
                s = re.sub("annotations/instances_train2017.json",
                           "VOC2007/ImageSets/Main/trainval.txt", s)
                s = re.sub("annotations/instances_val2017.json",
                           "VOC2007/ImageSets/Main/test.txt", s)
                s = re.sub("annotations/instances_val2017.json",
                           "VOC2007/ImageSets/Main/test.txt", s)
                s = re.sub("train2017", "VOC2007", s)
                s = re.sub("val2017", "VOC2007", s)
            else:
                s = re.sub("data_root = 'data/VOCdevkit/'",
                           "data_root = \'"+ outputPath[0:outputPath.rfind("/")] +"/VOC2007/\'", s)
                s = re.sub('img_prefix=.*?\],',
                           "img_prefix=data_root + '',".format(total_epochs), s)
        with open(fname, 'w') as f:
            f.write(s)
    

    def train(self, framework_path = None, n_gpus = 1):
        os.system("python mmdetection/tools/train.py " + self.config_fname)

        shutil.rmtree(os.path.join(self.OUTPUT_PATH, "VOC" + self.DATASET_NAME+"_"+str(self.model)))



    def evaluate(self):
        predictor = mmdetectionPredict(os.path.join('./work_dirs/'+self.model+'/latest.pth'),
            os.path.join(self.OUTPUT_PATH,self.DATASET_NAME, self.DATASET_NAME + "_classes.csv"),
            self.model,self.model+self.DATASET_NAME)

        map = Map(predictor, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

def main():
    pass

if __name__ == "__main__":
    main()
