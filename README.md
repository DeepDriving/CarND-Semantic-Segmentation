# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).


### Model
This project is based on Fully Convolutional Netowrks (FCN and the architecture is based on "Fully Convolutional Network for Semantic Segmentation" paper by Jonathon Long and et.al.  

The model is based on VGG 16 with additional features.  
First, we append a 1x1 convolution to layer 7 output and add a 2x upsampling layer.  Next, we fuse this output with the 1x1 convolution layer on top of layer 4 and add a 2x upsampling layer.  Last, we fuse this output with the 1x1 convolution alyer on top of layer 4 and add a 8x upsampling layer.

### Optimization
The network is trained using cross entropy loss as the metric to minimize.  Adam Optimizer is used to minimize the losss.

### Training
Hyperparameter selection:
Learning rate
Epochs
Batch Size
Drop out 

Tried different value of learning rate and Drop out rate.  

Learning rate = 0.0001 with drop out = 0.7 seems work pretty well.  The number of Epochs was selected by watching the convergence of the loss.  After 
### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
