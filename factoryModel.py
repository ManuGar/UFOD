from objectDetectors.MXNetObjectDetector import SSDMxnet
from objectDetectors.RCNNObjectDetector import RCNNDetector
from objectDetectors.RetinaNetObjectDetector import RetinaNetDetector
from objectDetectors.TensorflowObjectDetector import tensorflowDetector
from objectDetectors.YOLOObjectDetector import TinyYoloV3Detector, YoloV3Detector


FRAMEWORKS= {
    "Mxnet" : {"ssdVgg16" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"ssd_300_vgg16_atrous_custom"),
               "ssdVgg16_512" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"ssd_512_vgg16_atrous_custom"),
               "ssdResnet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"ssd_512_resnet50_v1_custom"),
               "ssdMobilenet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"ssd_512_mobilenet1.0_custom")
               },
    "Rcnn" : {"mask-rcnn" : RCNNDetector.RCNNDetector},
    "Retinanet" : {"retinanet" : RetinaNetDetector.RetinaNetDetector},
    "Tensorflow" : {"ssdInception" : lambda d,n,o : tensorflowDetector.TensorflowDetector(d,n,o,"ssd_inception_v2_coco"),
               "fasterRcnnResnet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"faster_rcnn_resnet50_coco"),
               "rfcnResnet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"rfcn_resnet101_coco"),
               "maskRcnnInception" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"mask_rcnn_inception_v2_coco")
               },
    "Darknet" : {"tinyYolo" : TinyYoloV3Detector.TinyYoloV3Detector,"yolo" : YoloV3Detector.YoloV3Detector},
}

# MODELS = {
#     "ssdMxnetResnet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"resnet"),
#     "ssdMxnetMobilenet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,o,"mobilenet"),
#     "rcnn" : RCNNDetector,
#     "retinanet" : RetinaNetDetector,
#     "tensorflow" : tensorflowDetector,
#
# }
#     "Tensorflow" : {"ssd" : tensorflowDetector.TensorflowDetector},

def createModel(framework, modelText, dataset, dataset_name):
    model = FRAMEWORKS[framework][modelText](dataset, dataset_name)
    return model

def main():
    pass

if __name__ == "__main__":
    main()