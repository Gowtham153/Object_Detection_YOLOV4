# tensorflow-yolov4-tflite
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

YOLOv4 Implemented in Tensorflow 2.0. 
Convert YOLO v4 .weights to .pb, .tflite format for tensorflow, tensorflow lite.


## Getting Started
Inorder to run this code we need python 3.7, pip, nvidia cuda 10.1, cudnn 7.5. After installing each of these packages add these paths to the environment system variables, this will help to run using windows terminal.
### Installing pip
Follow this link to install pip on windows
```
https://phoenixnap.com/kb/install-pip-windows
```
### Installing Nvidia Cuda
```
https://developer.nvidia.com/cuda-10.1-download-archive-update2
```
### Installing Nvidia Cudnn
You have to signup/login inorder to download, which is free for everyone.
```
https://developer.nvidia.com/rdp/cudnn-archive
```
### Setting up python path to run the code using windows terminal
```
C:\Users\gowth\AppData\Local\Programs\Python\Python37
```
### Setting up pip path to run using windows terminal
```
C:\Users\gowth\AppData\Local\Programs\Python\Python37\Scripts
```
###

### Installing all the requirements using pip

```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```



## Downloading Official Pre-trained Weights
YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes we will use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights



## YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 



# Run yolov4 tensorflow model
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/kite.jpg



# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/video.mp4 --output ./detections/results.avi



# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
```

### Result Image/Video

You can find the output image or video in the Detections folder.




#### Nvidia 1660 Ti (intel CORE i7 9th Gen)

| Detection   |  416x416 |
|-------------|----------|  
| YoloV4 FPS  |    22    |



The training performance is not fully reproduced yet, so I recommended to use Alex's [Darknet](https://github.com/AlexeyAB/darknet) to train your own data, then convert the .weights to tensorflow or tflite.





### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  * Forked: https://github.com/hunglc007/tensorflow-yolov4-tflite
   
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)

