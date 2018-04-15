[//]: # (Image References)

[image1]: ./example/cameras3.png "Cameras3"
[image2]: ./example/modelError.png "ModelPerf"




# **Behavioral Cloning for Autonomous Driving**  

I build a learner to drive autonomously a simulation bot. The learner tells the simulation bot the steering angle depending on the images captured by three cameras mounted on the bot.

The simulation bot is able to complete autonomously the first track of the Udacity simulator.

---
## Dependencies
* [Driving Simulator](https://github.com/udacity/self-driving-car-sim)
* Python 3.x
* NumPy
* OpenCV
* Random
* Pandas
* sklearn
* Keras 1.2

---
## Goals / Steps
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written

---
## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* This README.md summarizing the results
* model.py (or model.ipynb) containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 recording the bot completing the first track


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 resource/run5
```

Images of the ride are saved in the directory _resource/run5_. The _video.mp4_ is generated as:
```sh
python video.py resource/run5
cp resource/run5.mp4 video.mp4
```


#### 3. Submission code is usable and readable

The _model.py_ (alternatively, _model.ipynb_) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

In the file, a generator, named _dataGenerator(df, batch_size = 100, augment=True)_ is provided (line 139). The final model architecture is trained both using and not using the generator and both versions of the model are suitable. The model.h5 file contains data of the trained network that uses the _dataGenerator_ with argument _batch_size=32_, that basically means 32*3*2 = 192 images for each batch (3 cameras and each image flipped)


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture is defined in the definition _MyKerasArchitecture(inputShape, NewSyntax = False)_ in model.py, line 87.
It consists of a series of convolutional layers and fully connected layers. It includes RELU and MaxPool layers to introduce nonlinearity and the data is normalized using a Keras lambda layer. 


#### 2. Attempts to reduce overfitting in the model

Previous version of the model contained dropout layers in an attempt to reduce overfitting. However, the performance was not better. The final model uses Max Pool layers after activated each convolutional layer in order to reduce model complexity. 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 206).



#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I drive about three tracks (approximately 10 minutes driving) and collected 9100 images for each camera, for a total of 27300 images. Each of these 27300 images is flipped, yielding 54600 images of size (160, 320, 3)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a series of convolutional layers, that requires less parameters as they share parameters among connections, followed by a few fully connected layers. 

At each subsequent layer, I increase the number of filters. The initial layer has 3 channels (RGB). Subsequent layers have a number of filters that are multiple of 3, up until 81 filters. 

At each subsequent layer I reduce the picture size by applying cropping, convolution and pooling with "valid" padding (shrink the size of picture).

The final fully connected layers reduces the size from 8019 to 200, to 50, to 1. That is the steering angle. At each of those layer the RELU activation is applied.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (model.py, line 192). The training set is randomly sampled to be 2/3 of the sample.


Previous versions of the model were not successful completing the track. The vehicle bot failed in specific spots. I added additional convolutional layer and increased the size of the fully connected layer to be 200. Overall, it seemed that with a model having around 1500000 to 2500000 parameters, train on my sample of 2/3 * 54600 images, is be suitable to successfully complete the first track autonomously.


#### 2. Final Model Architecture

The final model architecture (model.py, lines 18-24) consisted of a convolution neural network with the following layers and layer sizes. The model has 10 layers, excluding image cropping and flattening layers.

| Layer (type)                     | Output Shape          | Param #     | Connected to            |        
|:--------------------------------:|:---------------------:|:-----------:|------------------------:|
| input_1 (InputLayer)             | (None, 160, 320, 3)   | 0           |                         |        
| lambda_1 (Lambda)                | (None, 160, 320, 3)   | 0           | input_1[0][0]           |        
| cropping2d_1 (Cropping2D)        | (None, 80, 320, 3)    | 0           | lambda_1[0][0]          |        
| convolution2d_1 (Convolution2D)  | (None, 75, 315, 9)    | 981         | cropping2d_1[0][0]      |        
| maxpooling2d_1 (MaxPooling2D)    | (None, 37, 157, 9)    | 0           | convolution2d_1[0][0]   |        
| convolution2d_2 (Convolution2D)  | (None, 32, 152, 27)   | 8775        | maxpooling2d_1[0][0]    |        
| maxpooling2d_3 (MaxPooling2D)    | (None, 16, 76, 27)    | 0           | convolution2d_2[0][0]   |         
| convolution2d_3 (Convolution2D)  | (None, 11, 71, 81)    | 78813       | maxpooling2d_3[0][0]    |         
| maxpooling2d_4 (MaxPooling2D)    | (None, 5, 35, 81)     | 0           | convolution2d_3[0][0]   |         
| convolution2d_4 (Convolution2D)  | (None, 3, 33, 81)     | 59130       | maxpooling2d_4[0][0]    |         
| flatten_1 (Flatten)              | (None, 8019)          | 0           | convolution2d_4[0][0]   |         
| dense_1 (Dense)                  | (None, 200)           | 1604000     | flatten_1[0][0]         |         
| dense_2 (Dense)                  | (None, 50)            | 10050       | dense_1[0][0]           |         
| dense_3 (Dense)                  | (None, 1)             | 51          | dense_2[0][0]           |         
|:--------------------------------:|:---------------------:|:-----------:|------------------------:|
| Total params: 1,761,800
| Trainable params: 1,761,800
| Non-trainable params: 0
|:--------------------------------:|:---------------------:|:-----------:|------------------------:|


The training and validation error for each epoch is as below:
![alt text][image2]



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving as much as possible. 

I then recorded driving another 2 tracks were I drive from the other way. Around the curves, I steer quite considerably (instead of braking) to recover the vehicle from left to right and then to the center (or from right to the left and then to the center of the lane). 

The following is an example image, captured while driving, from respectively left camera, central camera, and right camera, 
![alt text][image1]

To augment the training data, each image and its steering measurement is flipped. Totally I have a training sample of 2/3 * 54600 images.

The training dataset contained in folder _data/_ is not uploaded in this repository. For an alternative training set, it is possible to use the one [provided by Udacity](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)




---
## Resources
* Udacity Self-Driving Car [Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) 
* Udacity project assignment and template on [GitHub](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
* Udacity project [rubric](https://review.udacity.com/#!/rubrics/432/view)
* Udacity Driving Simulator on [GitHub](https://github.com/udacity/self-driving-car-sim)
* Training [Data](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)

