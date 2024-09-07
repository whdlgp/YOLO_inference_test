# YOLO_inference_test
YOLO inference test for OpenCV DNN modules

## How to use
### Build
* You need OpenCV build with CUDA, cuDNN
  * OpenCV 4.10.0
  * CUDA 12.4
  * cuDNN 9.3.0

```
git clone https://github.com/whdlgp/YOLO_inference_test.git
cd YOLO_inference_test
mkdir build
cd build
cmake ..
make
```
* Binary output will be located 'testspace' directory

### You can
* You can modify my main.cpp file and module files for your flavor
* Currently support models
  * YOLO v4 (Darknet)
  * YOLO v5
  * YOLO v6
  * YOLO v7 (Darknet)
  * YOLO v8
  * YOLO v9
  * YOLO v10 (Only support modified version, CPU inference)

## Download Converted Official YOLO models
* You can find here
https://huggingface.co/choyg/YOLO_ONNX_models

