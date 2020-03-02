import os
import shutil
import urllib.request
import tarfile
import re
import numpy as np

from objectDetectors.objectDetectionInterface import IObjectDetection
from objectDetectors.TensorflowObjectDetector.functions import PascalVOC2TensorflowRecords
from Predictors.TensorflowPredict import TensorflowPredict
import wget

from Evaluators.MapEvaluator import MapEvaluator as Map

# Number of training steps.
num_steps = 20000

# Number of evaluation steps.
num_eval_steps = 50


MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 1
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'batch_size': 8
    }
}

class TensorflowDetector(IObjectDetection):
    def __init__(self, dataset_path, dataset_name, model):
        super(TensorflowDetector, self).__init__(dataset_path,dataset_name)
        self.model = model
        # IObjectDetection.__init__(self, dataset_path, dataset_name)

    def transform(self):




        # Este lo tenemos que cambiar por las funciones del cuaderno para pasar de un formato a otro.



        # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]

        class_path = os.path.join(self.OUTPUT_PATH, self.DATASET, "classes.names")
        result_path = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME)
        file = open(class_path, "r")
        cl_txt = ""
        i = 1
        for cl in file:
            cl_txt += "item {\n\tid: " + str(i) + "\n\tname: '" + cl.split("\n")[0] + "'\n}\n"
            i += 1
        label_map = open(os.path.join(result_path, "label_map.pbtxt"), "w+")
        label_map.write(cl_txt)
        label_map.close()
        print(cl_txt)

        PascalVOC2TensorflowRecords(self.DATASET, self.OUTPUT_PATH)
        # shutil.rmtree(dataset_path)


    # En este caso tendremos que hacer primero la division del dataset para que luego la transformacion se haga
    # desde el dataset de entrenamiento y de evaluacion
    # def organize(self, train_percentage):
    #     # dataset_name = dataset_path[dataset_path.rfind(os.sep) + 1:]
    #     fn.datasetSplit(self.DATASET_NAME,self.DATASET,self.OUTPUT_PATH,train_percentage)

    def createModel(self):
        pass


    def train(self, framework_path= None, n_gpus = 1):
        MODEL = MODELS_CONFIG[self.model]['model_name']
        aux_path = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "annotations")
        test_record_fname = os.path.join(aux_path,"test.record")
        train_record_fname = os.path.join(aux_path,"train.record")
        label_map_pbtxt_fname = os.path.join(aux_path, "label_map.pbtxt")
        # Name of the pipline file in tensorflow object detection API.
        pipeline_file = MODELS_CONFIG[self.model]['pipeline_file']

        # Training batch size fits in Colabe's Tesla K80 GPU memory for selected model.
        batch_size = MODELS_CONFIG[self.model]['batch_size']
        MODEL_FILE = MODEL + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        DEST_DIR = self.OUTPUT_PATH

        if not (os.path.exists(MODEL_FILE)):
            urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

        tar = tarfile.open(MODEL_FILE)
        tar.extractall()
        tar.close()

        os.remove(MODEL_FILE)
        if (os.path.exists(DEST_DIR)):
            shutil.rmtree(DEST_DIR)
        os.rename(MODEL, DEST_DIR)
        fine_tune_checkpoint = os.path.join(DEST_DIR, self.model + "model.ckpt")
        pipeline_fname = os.path.join('/content/models/research/object_detection/samples/configs/', pipeline_file)

        assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)

        num_classes = self.get_num_classes(label_map_pbtxt_fname)
        with open(pipeline_fname) as f:
            s = f.read()
        with open(pipeline_fname, 'w') as f:

            # fine_tune_checkpoint
            s = re.sub('fine_tune_checkpoint: ".*?"',
                       'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

            # tfrecord files train and test.
            s = re.sub(
                '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
            s = re.sub(
                '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

            # label_map_path
            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

            # Set training batch_size.
            s = re.sub('batch_size: [0-9]+',
                       'batch_size: {}'.format(batch_size), s)

            # Set training steps, num_steps
            s = re.sub('num_steps: [0-9]+',
                       'num_steps: {}'.format(num_steps), s)

            # Set number of classes num_classes.
            s = re.sub('num_classes: [0-9]+',
                       'num_classes: {}'.format(num_classes), s)
            f.write(s)

        model_dir = 'training/'
        # Optionally remove content in output model directory to fresh start.
        os.remove(model_dir)
        os.makedirs(model_dir, exist_ok=True)



        os.system("/content/models/research/object_detection/model_main.py --pipeline_config_path=" +pipeline_fname +
                  "--model_dir=" + model_dir + " --alsologtostderr --num_train_steps=" + str(num_steps)+ " --num_eval_steps=" + str(num_eval_steps) )

        output_directory = os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "models",
                     self.model + '_' + self.DATASET_NAME + '_final.ckpt')
        # './fine_tuned_model'
        lst = os.listdir(model_dir)
        lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
        steps = np.array([int(re.findall('\d+', l)[0]) for l in lst])
        last_model = lst[steps.argmax()].replace('.meta', '')

        last_model_path = os.path.join(model_dir, last_model)
        print(last_model_path)
        os.system( "/content/models/research/object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=" +
                   pipeline_fname+ "--output_directory=" + output_directory+ " --trained_checkpoint_prefix=" + last_model_path)

    def evaluate(self):
        tensorflowPredict = TensorflowPredict(os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "models",
                                                 self.model + "_" + self.DATASET_NAME + "_final.params"),
                                    os.path.join(self.OUTPUT_PATH, self.DATASET_NAME, "annotations", "label_map.pbtxt"),
                                    self.model)

        map = Map(tensorflowPredict, self.DATASET_NAME, os.path.join(self.OUTPUT_PATH, self.DATASET_NAME), self.model)
        map.evaluate()

    def get_num_classes(self, pbtxt_fname):
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())