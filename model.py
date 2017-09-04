import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from random import shuffle, randint, random

images = []
measurements = []

lines = []

for folder in ['track_1_forward', 'track_1_backward']:
    print(folder)
    with open('data_for_carND/'+folder+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if -0.01 < float(line[3]) < 0.01 and random() < 0.5:
                continue
            lines.append(line)

import sklearn
from sklearn.model_selection import train_test_split
shuffle(lines)
# train_samples, validation_samples = train_test_split(lines)
train_samples = lines
validation_samples = []

def generator(samples, batch_size=32*4):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                corr = [0, 0.9, -0.9]
                line = batch_sample
                # ind = randint(0, 2)
                # name = batch_sample[ind]
                # center_image = cv2.imread(name.lstrip(' '))
                # center_angle = float(batch_sample[3]) + corr[ind]
                # # if randint(0, 1) == 1 and ind == 0:
                # #     center_image = cv2.flip(center_image, 1)
                # #     center_angle *= -1.0
                # imhsv = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                # imhsv[:,:,-1] = (imhsv[:,:,-1].astype('float32')*random()).astype('uint8')
                # center_image = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)
                # images.append(center_image)
                # angles.append(center_angle)

                dir_path = os.path.dirname(os.path.realpath(__file__))
                img_folder = line[0].split('\\')[5]
                path = dir_path + '/data_for_carND/' + img_folder + '/IMG/' + line[0].split('\\')[-1]
                img_center = cv2.imread(path)
                path = dir_path + '/data_for_carND/' + img_folder + '/IMG/' +line[1].split('\\')[-1]
                img_left = cv2.imread(path)
                path = dir_path + '/data_for_carND/' + img_folder + '/IMG/' +line[2].split('\\')[-1]
                img_right = cv2.imread(path)
                steer = float(line[3])
                images.append(img_center)
                measurements.append(steer)
                # images.append(cv2.flip(img_center, 1))
                # measurements.append(-1.0*steer)

                images.append(img_left)
                measurements.append(steer+corr[1])
                # images.append(cv2.flip(img_left, 1))
                # measurements.append(-1.0*steer-corr[1])

                images.append(img_right)
                measurements.append(steer+corr[-1])

                # images.append(cv2.flip(img_right, 1))
                # measurements.append(-1.0*steer-corr[-1])

            for center_image in images:
                imhsv = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                imhsv[:,:,-1] = (imhsv[:,:,-1].astype('float32')*random()).astype('uint8')
                center_image = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# X_train = np.array(images)
# y_train = np.array(measurements)

train_gen = generator(train_samples)
# validation_gen = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((80, 15), (0, 0))))
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(48, 3, 3, activation='relu', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
# model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
opt = Adam()
model.compile(loss='mse', optimizer=opt)
# model.fit_generator(train_gen, samples_per_epoch=len(train_samples)*6, validation_data=validation_gen, nb_val_samples=len(validation_samples)*6, nb_epoch=5)
if os.path.exists('model.h5'):
    print('loading model')
    model.load_weights('model.h5')
model.fit_generator(train_gen, samples_per_epoch=len(train_samples)*3, nb_epoch=30)
model.save('model.h5')
