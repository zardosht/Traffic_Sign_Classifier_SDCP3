## Traffic Sign Classification
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we create a convolutional neural networks to classify traffic signs. The model is trained and validated on [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

The description of the model and different steps of the classification pipline can be found [here](./writeup_template.md)

The requirements to pass the project are listed [here](./P3_Project_Rubric.pdf).


<!--
To meet specifications, the project will require submitting three files: 

* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 
-->

The Project
---
The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

<!--

### Dependencies
This lab requires:

* TensorFlow

-->

### Dataset 

1. Download the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)
2. Create a `./data/` folder next to the Jupyter notebooka, unzip the downloaded dataset file and put the `.p` files in the `./data/` folder. 

