# object_detector

**object_detector** is a program allowing to detect and localize object on a video using the YoloV3 model.
As this model as been trained on the coco dataset, the model can only detect class from this dataset.
You can find the complete list of the detetable classes in the file *resources/coco.names*.

## Installation

1) Create a virtualenv (optional)
Dependencies will be installed on a isolated Python environment which is often recommanded.
```bash
cd object_detector
python3 -m pip install --user virtualenv # install virtualenv
python3 -m venv venv # Create virtual env called "venv"
source venv/bin/activate # Activate virtual env
pip install --upgrade pip # Upgrade pip
cd ..
```

2) Install requirements
```bash
cd object_detector
pip install -r requirements.txt
```

3) Download Yolo weights in *resources/* folder
```bash
wget -O resources/yolov3.weights https://pjreddie.com/media/files/yolov3.weights # Download yolo weights
```

## Usage

In the python project:
```bash
python detect_objects.py -i <PATH_TO_INPUT_VIDEO> -c person
```
A video in avi format will be generate in the folder *detect_objects/generated*.

### Detect objects of multiple classes

Yolo can detect is not limited to detect objects of one class. You can detect as much classes as you want with the -c argument.

Example:
```bash
python detect_objects.py -i <PATH_TO_INPUT_VIDEO> -c person -c car
```
### Optional arguments:

* **-d --display-class-name**: Display the class name of each detected object. **Default False**

* **-t --detection-threshold**: Confidence threshold above which an object should be localized in the video. **Default 0.5**

## Editing config file

The file *resources/config.json* can be edited as follow:

* **default_detection_threshold**: the confidence threshold above which an object will be detected

## Performances

It takes **18 minutes** to process the following video ov 1125 frames of size ():

*https://www.youtube.com/watch?v=h4s0llOpKrU*

On a computer with the folowing characterisitcs:
* CPU: **i7-5500U CPU @ 2.40GHz**
* RAM: **16Gb**

## Next step

### Use of GPU

This version does not support using gpu yet. OpenCV 4.2.0 supports gpu for inference using openCV's dnn module improving hugely speed performance of models.


### Use of other models

Other architecture suchs as Single Shot Detectors (SSD) or Mask R-CNN are able to detect objects within a frame and it would be interesting to compare their performances.
If you are interested in such comparaisons you can find some answers in the official Yolov3 paper: *https://arxiv.org/pdf/1804.02767.pdf*
