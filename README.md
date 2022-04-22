# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization


An easy to use implementation of GradCam for Tensorflow models.

## Requirements:
1. Tensorflow model built using Functional API.
2. Docker and Nvidia Docker(GPU)


## How to use?
Building the container
`docker build -t env_name .`

Acessing the container (jupyter notebook)
`docker run --gpus all -v "$(pwd)":/code -p 8888:8888 env_name`

## Limitations:
The model should preferably be built using tensorflow:nightly-gpu-jupyter to avoid compatibility issues.


Using the module
```
from gradcam import GradCAM
#loading model and image size
gd = GradCAM('model.h5',size=(300,300))
#display
gd.display(img_path='test.jpg')
``` 