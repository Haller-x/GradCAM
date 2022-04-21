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

Using the module
```
from gradcam import GradCAM
#loading model and image size
gd = Gradcam('model.h5',size=(300,300))
#display
gd.pure_display(img_path='test.jpg')
``` 
