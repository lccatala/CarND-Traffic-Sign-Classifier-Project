# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataset_visualization]: ./images/dataset_visualization.png "Training set visualization"
[dataset_visualization2]: ./images/dataset_visualization2.png "Testing set visualization"
[dataset_visualization3]: ./images/dataset_visualization3.png "Validation set visualization"
[dataset_visualization4]: ./images/dataset_visualization4.png "Number of occurrences of each class"
[preprocessed_image]: ./images/preprocessed_image.png "Preprocessed image"
[new_0]: ./new_images/0.jpg "New image 0"
[new_1]: ./new_images/1.jpg "New image 1"
[new_2]: ./new_images/2.jpg "New image 2"
[new_3]: ./new_images/3.jpg "New image 3"
[new_4]: ./new_images/4.jpg "New image 4"
[new_5]: ./new_images/5.jpg "New image 5"
[new_6]: ./new_images/6.jpg "New image 6"
[new_7]: ./new_images/7.jpg "New image 7"
[new_8]: ./new_images/8.jpg "New image 8"
[new_9]: ./new_images/9.jpg "New image 9"
[top5]: ./images/probabilities1.jpg "Top 5 softmax probabilities for every image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lccatala/CarND-Traffic-Sign-Classifier-Project/)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. We start by drawing 4 random examples from each dataset (training, testing and validation)

![alt text][dataset_visualization]
![alt text][dataset_visualization2]
![alt text][dataset_visualization3]

Then we plot the number of occurrences of each class in each dataset

![alt text][dataset_visualization4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

We tried and combined several preprocessing techniques, namely:
* Grayscaling
* Feature Normalization
* Feature Rescaling
* Histogram Equalization

Out of these, we found out that Histogram Equalization and Feature Rescaling (in that order) achieved the best results. These operations are performed in the function `preprocess()`.

Here is an example of a traffic sign image after preprocessing.

![alt text][preprocessed_image]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

We started with a standard LeNet5 model. However, after some testing, we found it got stuck pretty early on (at around 60% accuracy without image preprocessing), so we added a couple of Dropout layers after each Fully Connected layer, one of the most common techniques to prevent a model from overfitting. We also tweaked `sigma` from 0.1 to 0.01.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 3x3	    |  1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flatten				|												|
| Fully connected		| Input = 400. Output = 120						|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| Input = 120. Output = 84						|
| Dropout				| 												|
| Fully connected		| Input = 84. Output = 43						|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

While experimenting with different parameter combinations, we found that `EPOCHS = 32`, `BATCH_SIZE = 128` and a learning rate `RATE = 0.001` achieved the best results. The batch size did not seem to affect the final model's accuracy, however the lower learning rate and higher epochs helped getting it over the required accuracy.

We found the actual epoch in which the model reached peak accuracy varies from training session to session, although it never happened over epoch 32. This is why se set it to that number and simply save the model with the highest accuracy found during training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.949
* test set accuracy of 0.933

* What was the first architecture that was tried and why was it chosen?
We started with LeNet5, since it is well known, easy to implement, has been used in similar scenarios (handwritten number classification) and was recommended by Udacity.

* What were some problems with the initial architecture?
It stopped improving quite early on, seamingly overfitting. Before we implemented the image preprocessing pipeline, which improved the final accuracy by itself, it only reached a validation accuracy of around 0.600. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Since we already faced a similar problem at a university course, we decided to add dropout layers to every fully connected layer, which prevented the model from focusing too much on specific pixels. This fixed the overfitting problem and allowed the model to reach a validation accuracy of 0.950 (with image preprocessing), which is greater than that required by the rubric.

* Which parameters were tuned? How were they adjusted and why?
We increased the number of epochs and decreased the learning rate, allowing for a smoother learning curve.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The combination of convolution and pooling layers was already proven effective by the original LeNet5 architecture. Also, we found out fully connected layers are usually combined with dropout layers to prevent overfitting, since they are the layers where most pixels could be considered and fixated over and, as said before, the second one would prevent the network from overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][new_0] ![alt text][new_1] ![alt text][new_2] ![alt text][new_3] ![alt text][new_4] ![alt text][new_5] ![alt text][new_6] ![alt text][new_7] ![alt text][new_8] ![alt text][new_9]

These images were taken as screenshots of driving videos such as [this one](https://www.youtube.com/watch?v=P47GIs3za00&t=1563s). They might be difficult to classify since not all of them are shown in their entirety (it's hard to capture a sign in 32 pixels), or they might be taken from too far away, making them too blurry. This last problem is especially prevalent in speed limit signs, where numbers can be mistakenly recognized.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        										| 
|:---------------------:|:-----------------------------------------------------------------:| 
| Priority road      			| Priority road   												| 
| Speed limit (60km/h)     		| Priority Road 										|
| Ahead only					| Ahead only												|
| Go straight or left	     	| Turn left ahead					 					|
| Keep right					| Keep right      													|
| Speed limit (60km/h)			| End of all speed and passing limits      					|
| Yield							| Yield      																|
| Roundabout mandatory			| Roundabout mandatory      								|
| No passing					| Yield      														|
| Wild animals crossing			| Wild animals crossing      								|




The model was able to correctly guess 5 of the 10 traffic signs, which gives an accuracy of 50%. This compares unfavorably against the accuracy on the test set, due to the problems mentioned before.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

The model is relatively sure of the class for every image except the 4th and 6th ones (with probabilities over 50%),although the second one is incorrectly predicted. 

The top five soft max probabilities for every image are

![alt text][top5]


