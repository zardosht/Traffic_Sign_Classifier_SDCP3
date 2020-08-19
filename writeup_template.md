#### Udacity Self-driving Car Nanodegree
# Project 3: Traffic Sign Classification
#### Using CNNs with Tensorflow 1.x



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[training_set_distribution]: ./writeup_images/training_set_distribution.png "Training Set Distribution"
[example_sample]: ./writeup_images/example_sample.png "An example training image"
[poor_sample]: ./writeup_images/poor_sample.png "A very dark sample"
[test_images]: ./writeup_images/test_images.png "Images for manual test"
[classification_results_on_test_images]: ./writeup_images/classification_results_on_test_images.png "Results of manual classification on test images"
[top5_probabilities1]: ./writeup_images/top5_probabilities1.png "Top 5 probabilities for each input image"
[top5_probabilities2]: ./writeup_images/top5_probabilities2.png "Top 5 probabilities for each input image"
[network_activations]: ./writeup_images/network_activations.png "Network activations"



## Rubric Points
The requirements to pass the project are listed [here](https://github.com/zardosht/Traffic_Sign_Classifier_SDCP3/blob/master/P3_Project_Rubric.pdf).
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zardosht/udacity_selfdriving_car/blob/master/P3_Traffic_Sign_Classifier/P3_Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

Following image shows a smaple image from the dataset. The images are of shape 32x32x3 RGB images with datatype `uint8` 

![alt text][example_sample]{height=400px width=500px}

There are 43 classes of German Traffic Signs. The dataset is not uniformly distributed over classes. For some classes there are more than 2000 sampels, whereas for some classes there are less than 200 samples. Follwing figure shows the distribution of samples over classes. 

![alt text][training_set_distribution]

Also the quality of sample images differs a lot. Some images have very good lighting, are clear and sharp, and occlussion free. Some samples are very dark, are partially occluded, have noise (like stickers attached on the sign). The following figure is an exmaple of a very dark sample, that is practically unrecognizable for human eye: 

![alt text][poor_sample]

**Summary of Insights on Data**

* Dataset is sequential (all images of a class are after each other)
* Images are 32x32x3 RGB, with dtype=np.uint8. 
* Images are cropped to the traffic sign. No environment. 
* The distribution of the training data is not uniform. With as few as about 180 images for classes like "0-Speed Limit (20 km/h)" or "19-Dangerous curve to the left", to around 2000 or more for classes like "3-Speed Limit (50 km/h)" or ""2-Speed Limit (30 km/h). 
* Quality of some images are not good. For example the first 25 images for the class "19-Dangerous curve to the left" are almost black, with nothing recognizable with human eyes. 





### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

Sinsce color is an important feature of traffic signs (the red ones signify dangers or prohibitions, the blue ones signify information) I decided to keep the RGB images for training and do not convert them to gray scale. 

My preprocessing only consists of normalizing the images to zero mean using the following formula: `(pixel - 128)/ 128`. This turns each color channel of the image into a `float32` array with values between -1 and 1. 

I also shuffled the training set before training. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with LeNet and adapted it for RGB images and number of calsses in the traffic sign database. 

| Layer         	               |     Description                                                       | 
|:-----------------------------:|:-------------------------------------------------------------:| 
| Input         	               | 32x32x3 RGB image   		                   	       | 
| Convolution 5x5x3x6   | 1x1 stride, valid padding, outputs 28x28x6        |
| RELU			       |									       |
| Max pooling	3x3         | 2x2 stride,  same paddding, outputs 14x14x6    |
| Convolution 5x5x6x16 | 1x1 strid, valid padding, outputs 10x10x16        |
| RELU                             |                                                                               |
| Max pooling	3x3         | 2x2 stride, same padding,  outputs 5x5x16         |
| Fully connected            | input 400, output 120                                           |
| Fully connected            | input 120, output 84                                             |
| Softmax (Output)         | input 84, output 43                                               |

Since I got the good results (according to the requirements of the project) I kept the architecture. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started with batch size 128, and 100 epochs. I then observed that after epoch 50 the validation set accuracy oscilates and does not imprve. So I reduced the epochs to 50. 
I then added dropout to the model, which imprved the validation set accuracy by about 2 percent (from almost 94% to 96%). 
Adam is used as optimizer for training the model. I started the training with learning rate 0.001. I wanted to apply learning rate decay, however read [here](https://stackoverflow.com/a/39526318/228965) that learning rate decay is not very effective for Adam since it adaptively associates an individual learning rated to every parameter in the model. The initial input learning rate is used as upper limit. 
Instead of reducing the leraning rate, I doubled the batch size after epoch 25, which should have an equivalent effect as described [here](https://arxiv.org/pdf/1711.00489.pdf)  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As said above, I started with LeNet. Becuase it is a simple model and I had already implemented it for the MNIST dataset. I adapted the model though for the RGB data (making the filters for the first CONV layer 5x5x3) and changed the output layer for 43 output classes of the traffice sign dataset. 

I think the decision not to convert the input into gray scale was a good one. Becuase I got 94% validation set accuracy right at the first try (with 100 epochs, batch size 128, and learning rate 0.001). I tuned the training by reducing the epochs to half, adding dropout to the model, and doubling the batch size after 25 epochs.   
Using these changes I achieved between 95.5 to 97 percent validation set accuracy in different training runs. 

The test set accuracy in the model is 96%. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I manually tested the model predications using the following images (15 images of German traffic signs collected from the Internet): 

![alt text][test_images]

Before feeding to classfier I converted the images to RGB (if not the case, e.g. PNG images) and resized them to 32x32 pixels with `LANCZOS` resampling. 

The result of manual predictions is shown in the figure below: 

![alt text][classification_results_on_test_images]




#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

As can be seen above the image 1 (filename: 6.jpg), a No Vehicle sign, is misclassified as 13-Yield. All other images are correctly classified. This gives an accuracy of (14/15)=0.93, wich is less than the test set accuracy (0.95). In the next question I discuss the performance of the model and possible reason for this misclassification.  


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The results of the top 5 probabilities of model for each input imge is shown in figures below (for all 15 images please refer to the [HTML export of the notebook](./) ). The classifier is 100% sure in most of the cases (porb=1.0). One image is misclassified: Image 6.jpg is a "15-No Vehicles" sign, but it is misclassified as a "13-Yield" sign. The second guess is however the correct class "15-No Vehicles". 

It is generally very intersting to look at the other guesses (other indices in top 5 probabilities). Most of the other guesses are classes that have visual resemblence to the target class. For example both classes "15-No Vehicle" and "13-Yield"  have empty white area inside the sign, and the sign is red and white color. This can be seen also in other exmples. For example image "14.jpg" which is correctly guessed as "Turn Left Ahead". The other guesses are '38-Keep right', '35-Ahead only', '37-Go straight or left' which all are round and visually similar sings showing white arrows on blue background. 

Looking at the distribution of the training data also gives insights into the misclassified case. The class "13-Yield" (wrong class) has almost 2000 training samples. However, the class "15-No Vehicle" (the correct class) less that 500 samples. Less training samples has lead to poor generaliztion for this class, specially considering the manual test image that has noise (the yellow Umleitung sign blow the No Vehicle sign). 

![alt text][top5_probabilities1]
![alt text][top5_probabilities2]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Figure below shows the network activations for the input image Speed Limit (30 km/h). As can be seen, right the first convolutional layer of the network reacts to round and high contrast patterns in the input image. 

![alt text][network_activations]

