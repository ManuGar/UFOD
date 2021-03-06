import  os
import glob
import wget
import shutil
from objectDetectors.objectDetectionInterface import IObjectDetection
from Predictors.RCNNPredict import RCNNPredict
from Evaluators.MapEvaluator import MapEvaluator as Map
from imutils import paths
from numpy import zeros, asarray
from os import listdir
from xml.etree import ElementTree
from objectDetectors.RCNNObjectDetector import functions as fn
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from pathlib import Path


class RCNNDetector(IObjectDetection):

    def __init__(self, dataset_path, dataset_name):
        super(RCNNDetector, self).__init__( dataset_path, dataset_name)
        self.train_set = ClassDataset()
        self.test_set = ClassDataset()
        # self.train_set = KangarooDataset()
        # self.test_set = KangarooDataset()
        self.model = "rcnn"
        self.modelWeights = None
        self.config = Config()

    def transform(self):
        # fn.organizeDataset(self.DATASET_NAME, self.OUTPUT_PATH, self.DATASET)
        self.train_set.load_dataset(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME), True)
        # self.train_set.load_dataset(dataset_path, True)
        self.train_set.prepare()
        #self.test_set.load_dataset(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME), False)
        # self.test_set.load_dataset(dataset_path, False)
        #self.test_set.prepare()

    # def organize(self, train_percentage):
    #     super(RCNNDetector, self).organize( train_percentage)

    def createModel(self):
        # En este caso tambien debe ser output por que ya se ha hecho la division y se ha guardado
        classes_file = os.path.join(self.OUTPUT_PATH,self.DATASET_NAME, "classes.names")
        file = open(os.path.join(classes_file))
        classes = []
        for line in file:
            classes.append(line)
        n_classes = fn.count_classes(classes)
        n_images = len(glob.glob(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"train/JPEGImages/*.jpg")))
        ClassConfig.NUM_CLASSES += n_classes
        ClassConfig.NAME = self.DATASET_NAME
        
        ClassConfig.N_IMAGES = n_images
        ClassConfig.STEPS_PER_EPOCH = n_images // (ClassConfig.GPU_COUNT * ClassConfig.IMAGES_PER_GPU)

        self.config = ClassConfig()
        # Por lo mismo de antes. El dataset ya esta procesado y guardado ahi. Es donde se tiene que trabajar con el en este caso
        # self.modelWeights = MaskRCNN(mode='training', model_dir=os.path.join(self.OUTPUT_PATH,"model"), config=self.config)
        if not os.path.exists(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models")):
            os.mkdir(os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models"))
        self.modelWeights = MaskRCNN(mode='training', model_dir=os.path.join(self.OUTPUT_PATH,self.DATASET_NAME,"models"), config=self.config)
        if not os.path.exists('objectDetectors/RCNNObjectDetector/mask_rcnn_coco.h5'):
            wget.download("https://www.dropbox.com/s/12ou730jt730qvu/mask_rcnn_coco.h5?dl=1", 'objectDetectors/RCNNObjectDetector/mask_rcnn_coco.h5')
        self.modelWeights.load_weights('objectDetectors/RCNNObjectDetector/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

    def train(self, framework_path = None, n_gpus = 1):
        ClassConfig.GPU_COUNT = n_gpus
        # self.model.train(self.TRAIN_SET, self.TEST_SET, learning_rate=self.CONFIG.LEARNING_RATE, epochs=5, layers='heads')
        self.modelWeights.train(self.train_set, self.train_set, learning_rate=self.config.LEARNING_RATE, epochs=5, layers='heads')
        results = []
        # Path(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "models")).rglob(".h5")
        for r in glob.glob(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME,"models","**","*5.h5" )):
            results.append(r)
        # results = [p for p in os.listdir(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME,"models")) if p.endswith(".h5") and "mask_rcnn_" + self.DATASET_NAME + "_0005" in p]
        shutil.copy2(results[0],os.path.join(self.OUTPUT_PATH, self.DATASET_NAME,"models","mask_rcnn_" + self.DATASET_NAME + "_0005.h5"))

    def evaluate(self):
        rcnnPredict = RCNNPredict(
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME,"models", "mask_rcnn_"+ self.DATASET_NAME.lower() + "_0005.h5"),
            os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "classes.names"))
        map = Map(rcnnPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

class ClassDataset(Dataset):
    # load the dataset definitions

    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        classes_file = open(os.path.join(dataset_dir, "classes.names"))
        # self.add_class("dataset", 1, classes_file[0].split("\n")[0])

        i = 1
        for cl in classes_file:
            self.add_class("dataset", i, cl.split("\n")[0])
            i += 1
        
        # define data locations
        images_dir_train = os.path.join(dataset_dir,"train", "JPEGImages/")
        annotations_dir_train = os.path.join(dataset_dir,"train", "Annotations/")
        images_dir_test = os.path.join(dataset_dir, "test", "JPEGImages")
        annotations_dir_test = os.path.join(dataset_dir, "test", "Annotations")

        # list_images = list(paths.list_files(os.path.join(dataset_dir, "dataset"), validExts=(".jpg")))
        # train_list = list(paths.list_files(os.path.join(dataset_dir, "train"), validExts=(".jpg")))
        # test_list = list(paths.list_files(os.path.join(dataset_dir, "test"), validExts=(".jpg")))

        # train_list, test_list, _, _ = train_test_split(list_images, list_images, train_size=percentage,random_state=5)

        if (is_train):
                for filename in listdir(images_dir_train):
                    imageid = filename[:-4]
                    img_path = images_dir_train + filename
                    ann_path = annotations_dir_train + imageid + '.xml'
                    self.add_image('dataset', image_id=imageid,path=img_path, annotation=ann_path)

        else:
                for filename in listdir(images_dir_test):
                    imageid = filename.split(os.sep)[-1][:-4]
                    img_path = os.path.join(images_dir_test, imageid + ".jpg")
                    ann_path = os.path.join(annotations_dir_test, imageid + ".xml")
                    self.add_image('dataset', image_id=imageid, path=os.path.abspath(img_path), annotation=os.path.abspath(ann_path))

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for objeto in root.findall('.//object'):
            for box in objeto.findall('.//bndbox'):
                xmin = int(box.find('xmin').text)
                ymin = int(box.find('ymin').text)
                xmax = int(box.find('xmax').text)
                ymax = int(box.find('ymax').text)
                coors = [xmin, ymin, xmax, ymax, objeto.find('name').text]
                boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(box[4]))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class ClassConfig(Config):
    NAME = "kangaroo"
    NUM_CLASSES = 1 
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # number of classes (background + kangaroo)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    N_IMAGES = 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 1000 // (GPU_COUNT * IMAGES_PER_GPU)

