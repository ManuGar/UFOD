# UFOD: A Unified Framework for Object Detection

UFOD is an open-source framework that enables the training and comparison of object detection models on 
custom datasets using different underlying frameworks and libraries. Currently, the most well-known object detection 
frameworks have been included in UFOD, and new tools can be easily incorporated thanks to UFOD's high-level API.


## Workflow of the UFOD framework

The workflow of UFOD, depicted in the following image, captures all the necessary steps to train several object detection 
models and select the best one. Such a workflow can be summarised as follows. First of all, the user selects the dataset 
of images and some configuration parameters (mainly, the algorithms that will be trained and the frameworks or libraries 
that provide them). Subsequently, UFOD splits the dataset into a training set and a testing set. The training set is
employed to construct several object detection models, and the best of those models is selected based on their 
performance on the testing set. The output of the framework is the best model, and an application, in the form of a
Jupyter notebook, to employ such a model. Apart from the first step --- that is, the selection of the models to be 
trained --- the rest of the process is conducted automatically by UFOD without any user intervention. 
A more detailed explanation of the main features of this framework can be found in our [draft paper](draft.pdf).


![workflow](images/DiagramUFOD.png)


## List of frameworks and libraries supported in UFOD

Currently, UFOD provides support for the following algorithms.


| Framework/Library | Algorithms supported |
|---------|----------------------|
| [Darknet](https://pjreddie.com/darknet/yolo/) | YOLO and TinyYOLO |
| [MXNet](https://gluon-cv.mxnet.io/)   | SSD with different backbones |
| [Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection) | SSD and Faster-RCNN |
| [RetinaNet for Keras](https://github.com/fizyr/keras-retinanet) | RetinaNet |
| [MaskRCNN for Keras](https://github.com/matterport/Mask_RCNN) | Mask RCNN |

## Installation






## Acknowledgments

This work was partially supported by Ministerio de Economía y Competitividad [MTM2017-88804-P], Agencia de Desarrollo 
Económico de La Rioja [2017-I-IDD-00018], a FPI grant from Community of La Rioja 2018 and the computing facilities of
Extremadura Research Centre for Advanced Technologies (CETA-CIEMAT), funded by the European Regional Development Fund 
(ERDF). CETA-CIEMAT belongs to CIEMAT and the Government of Spain.
