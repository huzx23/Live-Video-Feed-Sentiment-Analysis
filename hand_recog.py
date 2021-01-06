import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os

import matplotlib.pyplot as plt

num_classes=10
img_rows,img_cols=240,640
batch_size=8

train_folder=r'C:\Users\huzai\Desktop\Folder\University Docs\Project 2021\Project Code\Datasets\Gestures\train'
validation_folder=r'C:\Users\huzai\Desktop\Folder\University Docs\Project 2021\Project Code\Datasets\Gestures\validation'

## IMAGE AUGMENTATION -- artificially expand image dataset to give the model data it has not seen in training set ##
train_datagen = ImageDataGenerator(
					rescale=1./255
# 					rotation_range=30,
# 					shear_range=0.3,
# 					zoom_range=0.3,
# 					width_shift_range=0.4,
# 					height_shift_range=0.4,
# 					horizontal_flip=True,
# 					fill_mode='nearest'
                    )

validation_datagen = ImageDataGenerator(rescale=1./255)

## Using the flow_from_directory() method to load our data+set from the directory
## which is augmented and stored in the train_gen and validation_gen variables.
train_gen = train_datagen.flow_from_directory(
					train_folder,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_gen = validation_datagen.flow_from_directory(
							validation_folder,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

## BUILDING THE MODEL -- 2D CNN WITH 6 LAYERS ##

model = Sequential()
##Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1))) ##32 FILTERS##
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

##Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) ##64 FILTERS##
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

##Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal')) ##128 FILTERS##
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

##Block-4 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) ##256 FILTERS##
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

##Block-5 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) ##256 FILTERS##
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

##Block-6

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

##Block-7

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

##Block-8

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

## MONITOR VALIDATION LOSS ##

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## SAVE THE MODEL TO A FILE ##
checkpoint = ModelCheckpoint('HandDetectionModel.h5', 
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

## REDUCE LEARNING RATE IF LEARNING STAGNATES ##
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 17000
nb_validation_samples = 3000
epochs=25

history=model.fit_generator(
                train_gen,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_gen,
                validation_steps=nb_validation_samples//batch_size)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.draw()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.draw()

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing  import image
import cv2
import numpy as np

##  LOADING HAAR CASCADE TO DETECT FACE ON WEBCAM ##
face_classifier=cv2.CascadeClassifier(r'C:\Users\huzai\Desktop\Folder\University Docs\Project 2021\Project Code\palm.xml')
classifier = load_model(r'C:\Users\huzai\Desktop\Folder\University Docs\Project 2021\Project Code\HandDetectionModel.h5')

class_labels=['Palm','L','Fist','Fist Moved','Thumb','Index','Ok','Palm Moved','C','Down'] 
cap=cv2.VideoCapture(0)

## DISPLAY CODE ##
while True:
    ret,frame=cap.read()
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(240,640),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('Gesture Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
