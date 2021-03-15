# KCS_Blood_Cells

Final project for a subject named Communication Human-Machine at Department of Informatics, University of Rijeka.

__Author__:
* Anton Ilić

__Mentors:__
* _prof. dr. sc._ Ivo Ipšić
* _doc. dr. sc._ Miran Pobar

__Date:__
* January, 2020

__Language:__
* Python3

__Structure:__
* _image_processing.py_ - Image processing
* _KCS_Blood_Cells.ipynb_ - Image classification

__Project language:__ 
* Croatian

__Dataset:__

[Kaggle](https://www.kaggle.com/paultimothymooney/blood-cells)

__Assignment description:__

This project represents a demonstration of using Python3 as a tool for simple computer vision operations and convolutional neural network classification on a dataset which contains white blood cell images.

_image_processing.py_

* Used for locating a white blood cell area (Region of interest) based on the cell's color range. Region of interest is then cropped and saved on another location.
* Implements OpenCV library.

_KCS_Blood_Cells.ipynb_

* Used for cell classification by using Tensorflow package and Keras API for deep learning.
* Implemented on Google Colab service for quicker model creation by using Google's Tesla GPU.


__Running instructions:__ 

Dataset should be saved in the project folder.

```
python3 image_processing.py
```
