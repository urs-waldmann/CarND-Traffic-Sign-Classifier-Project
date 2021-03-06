# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
---

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. Here is a link to my [project code](https://github.com/urs-waldmann/CarND-Traffic-Sign-Classifier-Project). 

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of the training set is 34799.
* The size of the validation set is 4410.
* The size of the test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the training data set. It is a bar chart showing how the training data is distributed.

![Traffic sign distribution before data augmentation](figures/distr_b_aug.jpg)

Here we see three traffic sign examples of the training data set.

![Traffic sign example 1 from training data](figures/visualization.jpg)
![Traffic sign example 2 from training data](figures/vis_2.jpg)
![Traffic sign example 3 from training data](figures/vis_3.jpg)

### Design and Test a Model Architecture

#### 1. Grayscale

As a first step, I decided to convert the images to grayscale. As it turned out I get a slightly higher accuracy with the coloured images. On the one hand the German traffic signs may work even for colour blind people but on the other hand the color is a distinct property of the traffic signs. So I decided to stick with the coloured images.

Here are three examples of traffic sign images after grayscaling.

![Traffic sign example 1 after grayscaling](./figures/vis_gray.jpg)
![Traffic sign example 2 after grayscaling](./figures/vis_gray_2.jpg)
![Traffic sign example 3 after grayscaling](./figures/vis_gray_3.jpg)

#### 2. Normalization

As a last step, I normalized the image data with the min-max method because in that way the mean of the distribution of the pixel-values is zero and the standard deviation for each pixel-value is the same.

Here are three examples of traffic sign images after normalization.

![Traffic sign example 1 after normalization](./figures/vis_norm.jpg)
![Traffic sign example 2 after normalization](./figures/vis_norm_2.jpg)
![Traffic sign example 3 after normalization](./figures/vis_norm_3.jpg)

#### 3. Training data augmentation

I decided to generate additional training data because some traffic signs are underrepresentated in the original training data set as we saw in the section *Data Set Summary & Exploration* above. This means that the neural network will train and thus optimize its parameters for the overrepresented traffic signs in a better way while it will train and thus optimize its parameters for the underrepresented traffic signs in a worse way. This will have an impact on the accuracy of the test set and the five new German traffic signs. Let us therefore consider the case where within my five new German traffic signs are only traffic signs that in the original training data are underrepresented. In this case my accuracy in predicting the class id of these five new German traffic signs will be low. If I instead consider the case where within my five new German traffic signs are only traffic signs that in the original training data are overrepresented my accuracy in predicting these five new German traffic signs will be high. To balance this I augmented the training data set.

A good way to augment data is to consider methods that already exist in the original data set. In the original training data set the traffic signs are shown from different perspectives, they have different size and their brightness is different. That is why I choose to augment the original training data set using rotation and resize.

To add more data to the the training data set, I resize some of the original training data to a shape of (28, 28, 3) and pad the edges such that my image shape is again (32, 32, 3). Furthermore I rotate some images by -10° and some by 10°.

Another way to augment the training data is to change the brightness of the original training data set. I did not consider this in my implementation.

Here are three traffic sign examples of added images:

![Traffic sign example 1 after augmentation](./figures/vis_aug.png)
![Traffic sign example 2 after augmentation](./figures/vis_aug_2.png)
![Traffic sign example 3 after augmentation](./figures/vis_aug_3.png)

We notice that the traffic signs are as recognizable as the traffic signs of the original data.

After augmenting the original training data the distribution of the traffic signs looks like this:

![Traffic sign distribution after data augmentation](figures/distr_a_aug.jpg)

We note that the distribution of the traffic signs is more even.

Basic summary of the data set after augmentation:
* The size of the training set is now 57028.
* The size of the validation set remains 4410.
* The size of test set remains 12630.
* The shape of a traffic sign image remains (32, 32, 3).
* The number of unique classes/labels in the data set remains 43.

### Model Architecture

#### 1. Final model architecture

My final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 normalized image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x6   |
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 14x14x11 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x6     |
| Fully connected		| inputs 400, outputs 120      					|
| RELU					|												|
| Dropout				| keep 50% of values 							|
| Fully connected		| inputs 120, outputs 84      					|
| RELU					|												|
| Output        		| inputs 84, outputs 43      					|


#### 2. Train the model

To train the model, I used the Adam optimizer. I have 10 epochs, a batch size of 64 and my learning rate is 0.001. Furthermore are the weights and biases variables initialized with a mean of zero and a standard deviation of 0.05.

#### 3. Discussion

My final model results are:
* Validation set accuracy of 94.3%.
* Test set accuracy of 92.9%.

I took an iterative approach to find the best solution for the model architecture:
* I chose the [LeNet](http://yann.lecun.com/exdb/lenet/) architecture because it is a well known architecture of a Convolutional Neural Network.
* There were regularization problems, i.e. overfitting.
* I adjusted the LeNet architecure by adding an additional 1x1 convolution between the two convolutional layers to make my network deeper. Furthermore I added dropout between the two fully connected layers due to overfitting.
* I tuned the amount of values to keep in dropout due to overfitting. I also tuned the filter size of the convolutions. The filter size souldn't be too big because this reduces the size of the model and thus leads to a reduction in accuracy. I furthermore tuned the filter depth in order to make the network deeper. It is more efficient to make the nework deeper than making it wider beause in deeper layers the network already recognizes more complicated structures.
* The advantages of Convolutional Neural Networks is that they are designed to recognize visual patterns directly from pixel images with minimal preprocessing. They can recognize patterns with extreme variability and with robustness to distortions and simple geometric transformations (rotation, resize). An additional 1x1 convolution helps to make the network deeper without changing the structure. Dropout prevents overfitting by setting half of the activations to zero. Thus the model has to learn in a way where it can never rely on any given activation to be present.

Discussing the LeNet architecture:
* I chose the [LeNet](http://yann.lecun.com/exdb/lenet/) architecture.
* It is a well known Convolutional Neural Network architecture for supervised classification problems.
* With the modified LeNet architecture I do not have regularization problems anymore. In addition I obtain a high validation and test accuracy of 94.3% and 92.9% respectively.
 
### Test the Model on New Images

#### 1. Choose five new German traffic signs

Here are five German traffic signs that I found on the web:

![New German traffic sign 4](./new-german-traffic-signs/vorfahrtsstrasse.jpg)
![New German traffic sign 3](./new-german-traffic-signs/hoechstgeschw_70.jpg)
![New German traffic sign 5](./new-german-traffic-signs/zwanzig.jpg)
![New German traffic sign 2](./new-german-traffic-signs/geradeaus.jpg)
![New German traffic sign 1](./new-german-traffic-signs/einf_verboten.jpg)

The second image might be difficult to classify because the angle of the picture of the traffic sign is much bigger than of the other training signs.
The last image will be very difficult to classify because it contains two signs. They are of the same type but they are very small. I added this image because I really want to see how well the trained network does on quite big changes.

#### 2. Predictions & discussion

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Speed limit (70km/h)			| Yield										|
| Speed limit (20km/h)					| Speed limit (20km/h)											|
| Ahead only	      		| Ahead only					 				|
| No entry			| Pedestrians      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set of 92.9%. Without the last image (that was really hard to predict) though the accuracy is 75%. This is closer to the test accuracy. Adding another fifth image that is easy to predict will most probably lead to an accuracy of 80%. This is close to the test accuracy.
I am surprised nevertheless that I did not get my first four new signs right. I was expecting a better accuracy of the new images.

#### 3. Quality of the model

The code for making predictions on my final model is located in the 22nd cell of my Ipython notebook.

For the first image, the model is absolutely sure that this is a priority road sign (probability of 1.0), and the image does contain a priority road sign. The top five softmax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999988         			| Priority road   									| 
| 6.62758e-06     				|End of no passing by vehicles over 3.5 metric ton					|
| 4.01964e-06					| End of no passing											|
| 5.18654e-07	      			| No entry					 				|
| 1.22818e-07 				    | Right-of-way at the next intersection      							|


For the second image, the model is relatively sure that this is a yield sign (probability of 0.6), but the image does contain a speed limit (70km/h) sign. The top five softmax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.591793         			| Yield   									| 
| 0.141179     				|No passing					|
| 0.114104					| Speed limit (120km/h)										|
| 0.0978724	      			| Speed limit (70km/h)					 				|
| 0.0414736 				    | No vehicles     							|

Only with a probability of 10% does the model predict the correct sign, i.e. a speed limit (70km/h). By augmenting the speed limit (70km/h) signs in the training data using a rotation method I can increase probability to predict this traffic sign correctly.

For the third image, the model is quite sure that this is a speed limit (20km/h) sign (probability of 0.9), and the image does contain a speed limit (20km/h) sign. The top five softmax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.883512         			| Speed limit (20km/h)  									| 
| 0.114056     				|Speed limit (30km/h)					|
| 0.000906641					| Speed limit (70km/h)									|
| 0.000687374	      			| No vehicles					 				|
| 0.000587383 				    | Speed limit (120km/h)     							|

For the fourth image, the model is absolutely sure that this is an ahead only sign (probability of 1.0), and the image does contain a ahead only sign. The top five softmax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999914        			| Ahead only  									| 
| 7.19175e-05    				|Turn left ahead					|
| 1.19258e-05 					| Turn right ahead									|
| 1.59193e-06	      			| Yield					 				|
| 2.53465e-08				    | Speed limit (60km/h)     							|

For the fifth image, the model is not really sure whether this is a pedestrians sign (probability of 0.5), and in fact the image does contain two no entry signs. The top five softmax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.49037        			| Pedestrians 									| 
| 0.313619   				|General caution					|
| 0.135102 					| Priority road									|
| 0.0153313 	      			| Road narrows on the right					 				|
| 0.0135095				    | Right-of-way at the next intersection     							|

This image was very difficult to predict since it contains two traffic signs (of the same type) that are relatively small on the image. Increasing the depth of the neural network can maybe improve this problem. Depth reduces the irrelevant spatial information of the image, i.e. the position of the sign on the image, and concentrates on the relevant information, i.e. typ of the traffic sign.
