from objectDetectors.MXNetObjectDetector import SSDMxnet
from objectDetectors.RCNNObjectDetector import RCNNDetector
from objectDetectors.RetinaNetObjectDetector import RetinaNetDetector
from objectDetectors.TensorflowObjectDetector import tensorflowDetector
from objectDetectors.YOLOObjectDetector import TinyYoloV3Detector, YoloV3Detector




MODELS = {
    "ssdMxnet" : SSDMxnet,
    "rcnn" : RCNNDetector,
    "retinanet" : RetinaNetDetector,
    "tensorfow" : tensorflowDetector,
    "tinyYolo" : TinyYoloV3Detector,
    "yolo" : YoloV3Detector

}

def createModel(modelText):
    return MODELS[modelText]

def main():
    pass

if __name__ == "__main__":
    main()