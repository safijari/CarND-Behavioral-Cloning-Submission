**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

---
###Files Submitted & Code Quality

#### 1. Project Files
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md is the writeup
* A video showing this model driving the car [here](https://www.youtube.com/watch?v=bFz6EO0N1R0&feature=youtu.be)

#### 2. Included Model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. The NVIDIA Model

My model is defined from lines 98 to 113. The first two layers preprocess the image (changing the range from 0-255 to -0.5 to 0.5, and cropping). Then I have four convoltuional layers, first two 5x5 and second two 3x3. All have stride of 2 and relu activation. I then flatten this, and stack five dense layers with 500, 100, 50, 10, and 1 output respectively. I intermix some dropout layers here to help fight overfitting.

#### 2. Attempts to reduce overfitting in the model

I added dropout, I also augmented the data with varying brightness to help fight overfitting.

I actually did not create a validation set for this project. I feel the input data in many cases is not a good representation of how I want the model to actually behave (more on this later), so the validation loss, in my opinion, wouldn't have been very useful. Instead the true test of the model was just running it on the simulator, which I did after every epoch.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I did augment the data using the left and right camera images and the correction to steering angle for those images sort of became a hyperparameter. Choosing a steering angle of 0.1 helped create a model that steered really smoothly, but it got confused at a turn and would cause the car to turn into a dirt path (which was safe, and the car would drive fine inside it ...). I eventually discovered that making the corrections for the side camera images really large (between 0.9 and 1.5) caused the model to learn to reeeeally avoid the sides of the road. This created erratic steering angle trajectories that kept going back and forth between large negative and positive values, but the average behavior was pretty smooth and the car drove fine.

#### 4. Appropriate training data

Really all I did was run the car forward 3 laps and backwards 3 laps, using the joystick. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose a proven model, I took NVIDIA's model. Optimized the training set over the MSE, no validatioin set. I used the simulator to guage how well the model was working.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I've already mentioned this.

#### 3. Creation of the Training Set & Training Process

I've already mentioned this, though one extra thing was to randomly discard images with small steering angles, to keep the model from becoming biased towards going straight.
