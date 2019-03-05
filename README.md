# Tensor Flow implementation of FCN for liquid detection in cups and glasses.

This code was developed as part of my Master Thesis with the title "[Detection of Liquids Using CNN for Robot Waiters](https://drive.google.com/open?id=1KtrCyWoIZaVFq6nq7Oi2qU-Iwi448hCt)". It is an extension of a normal VGG-16 fully convolutional network with two streams for detection of water. One stream corresponds to RGB frames and the second is for near infrared (NIR) frames. For a thorough description of this code please refer to Chapter 4, including the way it was trained and tested.

** Note that this code differs from the thesis by further convolving the layers after the two streams were merged.

![Result](cups.gif)

The corresponding video can be found [here](https://www.youtube.com/watch?v=AEJffBjgdXY). 


## Dataset
You can download the dataset used for this project [here](https://drive.google.com/open?id=19D2r6SOjK9edD9cSja3GCpKsDMSqKO9F). 

The dataset consists of 25,566 images: RGB, NIR, Thermal, Ground truth Label and contour.

* RGB: images of one or two glasses filled with different levels of water or empty. Used for training.
* NIR: images captured with a near infrared sensor of the same scene represented in the corresponding RGB frame. Used for training.
* Thermal: images used for acquiring the ground truth labels by thresholding the yellow, light pixels. Not used for training.
* Ground truth label: pixels colored in blue represent water from the thermal frame. Used for training.
* Contour: graphic image of the contour of the label over the RGB frame. Not used for training.

The directories are divided into: test, test_labels, train, train_labels, val, val_labels.

## Instructions

1 - Download the dataset on the link above and extract it under a directory named 'data'.

2 - Make sure to change the directories in train_nirct_mm.py to match your data structure.

3 - Adjust the hyperparameters on line 35 of train_nirct_mm.py as you please. The default number of epochs is set to 60, but 20 give good results as well (and finishes in a third of the time).

4 - Change the 'model_name' in line 50 to a name of your choosing.

5 - Run the program by using the following command in the command line:
```sh
$ python3 train_nirct_mm.py
```
6 - The program will run and give you the results in a new directory under a directory named 'runs' created automatically.

## License

MIT License.
