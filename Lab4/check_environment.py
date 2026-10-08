"""Check the Pi packages, model files and camera command."""
import platform
import shutil
import cv2
import numpy
import PIL
from common import configure,load_model
configure();load_model()
print('Python',platform.python_version(),'OpenCV',cv2.__version__,'NumPy',numpy.__version__,'Pillow',PIL.__version__)
print('rpicam-vid:',shutil.which('rpicam-vid') or 'not found (required on the Pi)')
print('YOLOv2 weights loaded.')
