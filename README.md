# UFOD: A Unified Framework for Object Detection

UFOD is an open-source framework that enables the training and comparison of object detection models on 
custom datasets using different underlying frameworks and libraries. Currently, the most well-known object detection 
frameworks have been included in UFOD, and new tools can be easily incorporated thanks to UFOD's high-level API.

## Information about UFOD

### Workflow of the UFOD framework

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


### List of frameworks and libraries supported in UFOD

Currently, UFOD provides support for the following algorithms.


| Framework/Library | Algorithms supported |
|---------|----------------------|
| [Darknet](https://pjreddie.com/darknet/yolo/) | YOLO and TinyYOLO |
| [MXNet](https://gluon-cv.mxnet.io/)   | SSD with different backbones |
| [Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection) | SSD and Faster-RCNN |
| [RetinaNet for Keras](https://github.com/fizyr/keras-retinanet) | RetinaNet |
| [MaskRCNN for Keras](https://github.com/matterport/Mask_RCNN) | Mask RCNN |

## Using UFOD

### Dependencies and installation

### Configuration




### How to launch the training process

Once the user has prepared the dataset and fixed the training options in the ```config.json``` file, the training process is launched using the following command:

```bash
python3 taskLauncher.py -c config.json
```

### Adding new frameworks and algorithms

UFOD has been designed to easily include new frameworks and object detection algorithms on it. To this aim, it is necessary to 
implement a new class. This process can be summarised as follows. Independently of the deep learning framework and algorithm used, the procedure to train an object detection model consists of the following steps: 

1. Organise the dataset using a particular folder structure;
2. Transform the annotations of the dataset to the correct format;
3. Create the necessary configuration files; and
4. Launch the training process. 

Steps 1, 2 and 4 are framework operations, since all the models of a framework require the same folder structure, annotation format, and their training process is launched in the same way; whereas, Step 3 depends on the concrete model. These steps have been modelled using an abstract class called ```IObjectDetection``` (see the following class diagram), that is particularised for each concrete framework and model. To be more concrete, for each framework that provides several detection algorithms, an abstract class providing the functionality for Steps 1, 2 and 4 is implemented (see, for instance, the class Darknet in the figure below); in addition, such a class is extended with subclasses for each individual model implementing the functionality for Step 3 (see the classes Tiny YOLOV3 and YOLOV3). In the case, of libraries that implement a single algorithm, a class with all the methods must be created (see, for instance, the class RetinaNet the below figure). Including a new framework or library in UFOD is as simple as defining a new class that extends ```IObjectDetection```. You can see examples of the implementations of these classes in the [objectDetectors](objectDetectors) folder of this repository. Once the class for the new framework/algorithm is defined, it is necessary to include it in the [factoryModel.py](factoryModel.py) file to facilitate its use. 

![workflow](images/DiagramUFOD2.png)

### Adding new evaluation metrics 

Each framework is able to compute the performance of their models; however, different frameworks employ different evaluation metrics; and, therefore, it is difficult to compare the models produced with different tools. To deal with this problem in UFOD, we have included a procedure to evaluate algorithms independently of the underlying framework. Such a procedure is a two-step process that can be summarised as follows. Given a folder with the images and annotations that will be employed for testing a model:

1. The model detects the objects in those images and stores the result in the Pascal VOC format inside the same folder; and
2. the original annotations and the generated detections are compared with a particular metric. 

This two-step process has been modelled in the UFOD API (see the figure below) using the abstract class ```IPredictor``` to implement the first step --- this class is particularised for each framework since all the models of the same framework perform predictions in the same way --- and the abstract class ```IEvaluator``` to implement the second step --- this class is particularised with different evaluation metric; and, currently, metrics such as the mAP, the IoU or the F1-score are supported. Using this approach, the models constructed using the UFOD framework can be compared even if they were constructed using different frameworks. Moreover, this part of the UFOD API can be employed to compare models constructed outside the framework by using a common metric. 

![workflow](images/DiagramUFOD3.png)


## Acknowledgments

This work was partially supported by Ministerio de Economía y Competitividad [MTM2017-88804-P], Agencia de Desarrollo 
Económico de La Rioja [2017-I-IDD-00018], a FPI grant from Community of La Rioja 2018 and the computing facilities of
Extremadura Research Centre for Advanced Technologies (CETA-CIEMAT), funded by the European Regional Development Fund 
(ERDF). CETA-CIEMAT belongs to CIEMAT and the Government of Spain.
