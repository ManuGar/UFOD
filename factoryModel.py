# FRAMEWORKS= {
#     "Mxnet" : {"ssdVgg16" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,"ssd_300_vgg16_atrous_custom"),
#                "ssdVgg16_512" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,"ssd_512_vgg16_atrous_custom"),
#                "ssdResnet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,"ssd_512_resnet50_v1_custom"),
#                "ssdMobilenet" : lambda d,n,o : SSDMxnet.SSDMxnet(d,n,"ssd_512_mobilenet1.0_custom")
#                },
#     "Rcnn" : {"mask-rcnn" : RCNNDetector.RCNNDetector},
#     "Retinanet" : {"retinanet" : RetinaNetDetector.RetinaNetDetector},
#     "Tensorflow" : {"ssdInception" : lambda d,n : tensorflowDetector.TensorflowDetector(d,n,"ssd_inception_v2_coco"),
#                "fasterRcnnResnet" : lambda d,n : tensorflowDetector.TensorflowDetector(d,n,"faster_rcnn_resnet50_coco"),
#                "rfcnResnet" : lambda d,n: tensorflowDetector.TensorflowDetector(d,n,"rfcn_resnet101_coco"),
#                "maskRcnnInception" : lambda d,n : tensorflowDetector.TensorflowDetector(d,n,"mask_rcnn_inception_v2_coco")
#                },
#     "Darknet" : {"tinyYolo" : TinyYoloV3Detector.TinyYoloV3Detector,"yolo" : YoloV3Detector.YoloV3Detector},
# }

def createModel(framework, modelText, dataset, dataset_name):
    model = ""
    if framework == "mmdetection":
        from objectDetectors.mmdetectionDetector import mmdetectionDetector
        model = mmdetectionDetector.mmdetectionDetector(dataset, dataset_name, modelText)
    if framework == "Efficientdet":
        from objectDetectors.EfficientDetDetector import EfficientDetDetector
        model = EfficientDetDetector.EfficientDetDetector(dataset, dataset_name, int(modelText))
    if framework == "FCOS":
        from objectDetectors.FcosDetector import FcosDetector
        model = FcosDetector.FcosDetector(dataset, dataset_name, modelText)
    if framework == "FSAF":
        from objectDetectors.FSAFDetector import FSAFDetector
        model = FSAFDetector.FSAFDetector(dataset, dataset_name, modelText)
    if framework == "Mxnet":
        from objectDetectors.MXNetObjectDetector import SSDMxnet
        if modelText == "ssdVgg16":
            aux = lambda d,n : SSDMxnet.SSDMxnet(d,n,"ssd_300_vgg16_atrous_custom")
            model = aux(dataset, dataset_name)
        elif modelText == "ssdVgg16_512":
            aux = lambda d, n: SSDMxnet.SSDMxnet(d, n, "ssd_512_vgg16_atrous_custom")
            model = aux(dataset, dataset_name)
        elif modelText == "ssdResnet":
            aux = lambda d, n: SSDMxnet.SSDMxnet(d, n, "ssd_512_resnet50_v1_custom")
            model = aux(dataset, dataset_name)
        elif modelText == "ssdMobilenet":
            aux = lambda d, n: SSDMxnet.SSDMxnet(d, n, "ssd_512_mobilenet1.0_custom")
            model = aux(dataset, dataset_name)
    elif framework == "Rcnn":
        from objectDetectors.RCNNObjectDetector import RCNNDetector
        if modelText == "mask-rcnn":
            model = RCNNDetector.RCNNDetector(dataset, dataset_name)
    elif framework == "Retinanet":
        from objectDetectors.RetinaNetObjectDetector import RetinaNetDetector
        if modelText == "retinanet":
            model = RetinaNetDetector.RetinaNetDetector(dataset, dataset_name)
    elif framework == "Tensorflow":
        from objectDetectors.TensorflowObjectDetector import tensorflowDetector
        if modelText == "ssdMobilenet":
            aux = lambda d,n : tensorflowDetector.TensorflowDetector(d,n,"ssd_mobilenet_v2")
            model = aux(dataset, dataset_name)
        elif modelText == "fasterRcnnInception":
            aux = lambda d, n: tensorflowDetector.TensorflowDetector(d, n, "faster_rcnn_inception_v2")
            model = aux(dataset, dataset_name)
        elif modelText == "rfcnResnet":
            aux = lambda d, n: tensorflowDetector.TensorflowDetector(d, n, "rfcn_resnet101")
            model = aux(dataset, dataset_name)
    elif framework == "Darknet":
        from objectDetectors.YOLOObjectDetector import TinyYoloV3Detector, YoloV3Detector
        if modelText == "tinyYolo":
           model = TinyYoloV3Detector.TinyYoloV3Detector(dataset, dataset_name)
        if modelText == "yolo":
           model = YoloV3Detector.YoloV3Detector(dataset, dataset_name)
    # model = FRAMEWORKS[framework][modelText](dataset, dataset_name)
    return model

def main():
    pass

if __name__ == "__main__":
    main()
