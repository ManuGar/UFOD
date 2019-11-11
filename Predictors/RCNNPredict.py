from Predictors.IPredictor import IPredictor
# USAGE
# python predict_batch.py --input logos/images --output output
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths



class RCNNPredict(IPredictor):
    CONFIDENCE = 0.5

    def __init__(self, modelWeights, classesFile):
        super().__init__(modelWeights, classesFile)
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def predict(self, imagePaths):
        # define the model
        testConfig = TestConfig()
        testConfig.NUM_CLASSES = 1+len(self.classes)
        rcnn = MaskRCNN(mode='inference', model_dir='./', config=testConfig)
        # load coco model weights
        # J. modificar con el path al modelo.
        rcnn.load_weights(self.modelWeights, by_name=True)

        imagePaths = list(paths.list_images(imagePaths))
        # loop over the input image paths
        for (i, imagePath) in enumerate(imagePaths):
            # load the input image (in BGR order), clone it, and preprocess it
            # print("[INFO] predicting on image {} of {}".format(i + 1,
            #	len(imagePaths)))

            # load the input image (in BGR order), clone it, and preprocess it
            img = load_img(imagePath)
            img = img_to_array(img)
            (hI, wI, d) = img.shape

            # detect objects in the input image and correct for the image scale
            # Poner short=512
            print(rcnn.config.NUM_CLASSES)
            print("22222222222222222222222222222222222222222222222222")

            results = rcnn.detect([img], verbose=1)
            r = results[0]
            boxes1 = []
            for (box, score, cid) in zip(r['rois'], r['scores'], r['class_ids']):
                if score < self.CONFIDENCE:
                    continue
                # Añadir label que sera con net.classes[cid]
                boxes1.append(([self.classes[cid - 1], box], score))

            # parse the filename from the input image path, construct the
            # path to the output image, and write the image to disk
            # filename = imagePath.split(os.path.sep)[-1]
            # outputPath = os.path.sep.join([args["output"], filename])
            file = open(imagePath[0:imagePath.rfind(".")] + ".xml", "w")
            file.write(self.generateXML(imagePath[0:imagePath.rfind(".")], imagePath, wI, hI, d, boxes1))
            file.close()

    def prettify(self,elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def generateXML(self,filename, outputPath, w, h, d, boxes):
        top = ET.Element('annotation')
        childFolder = ET.SubElement(top, 'folder')
        childFolder.text = 'images'
        childFilename = ET.SubElement(top, 'filename')
        childFilename.text = filename[0:filename.rfind(".")]
        childPath = ET.SubElement(top, 'path')
        childPath.text = outputPath + "/" + filename
        childSource = ET.SubElement(top, 'source')
        childDatabase = ET.SubElement(childSource, 'database')
        childDatabase.text = 'Unknown'
        childSize = ET.SubElement(top, 'size')
        childWidth = ET.SubElement(childSize, 'width')
        childWidth.text = str(w)
        childHeight = ET.SubElement(childSize, 'height')
        childHeight.text = str(h)
        childDepth = ET.SubElement(childSize, 'depth')
        childDepth.text = str(d)
        childSegmented = ET.SubElement(top, 'segmented')
        childSegmented.text = str(0)
        # boxes tiene que contener labels
        for (box, score) in boxes:
            # Cambiar categoria por label
            category = box[0]
            box = box[1].astype("int")
            #######
            # Cuidado esto está cambiado con respecto a lo que es habitualmente
            #######
            (y, x, ymax, xmax) = box
            childObject = ET.SubElement(top, 'object')
            childName = ET.SubElement(childObject, 'name')
            childName.text = category
            childScore = ET.SubElement(childObject, 'confidence')
            childScore.text = str(score)
            childPose = ET.SubElement(childObject, 'pose')
            childPose.text = 'Unspecified'
            childTruncated = ET.SubElement(childObject, 'truncated')
            childTruncated.text = '0'
            childDifficult = ET.SubElement(childObject, 'difficult')
            childDifficult.text = '0'
            childBndBox = ET.SubElement(childObject, 'bndbox')
            childXmin = ET.SubElement(childBndBox, 'xmin')
            childXmin.text = str(x)
            childYmin = ET.SubElement(childBndBox, 'ymin')
            childYmin.text = str(y)
            childXmax = ET.SubElement(childBndBox, 'xmax')
            childXmax.text = str(xmax)
            childYmax = ET.SubElement(childBndBox, 'ymax')
            childYmax.text = str(ymax)
        return self.prettify(top)


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGES_PER_GPU = 1
    ##### J. Esto hay que cambiarlo dependiendo de cada problema
    NUM_CLASSES = 0#1 + len(RCNNPredict.classes)









# cv2.imwrite(outputPath, output)