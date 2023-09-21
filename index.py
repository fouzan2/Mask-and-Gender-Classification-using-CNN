# first we train a model for mask classification
import os
from keras.preprocessing.image import ImageDataGenerator
train_dir = os.path.join('mask/Face Mask Dataset', 'Train')
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),class_mode='categorical')

validation_dir = os.path.join('mask/Face Mask Dataset', 'Validation')
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150, 150),class_mode='categorical')

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(225, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))

from keras import optimizers
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

history = model.fit_generator(train_generator,epochs=10,validation_data=validation_generator)

import joblib
joblib.dump(model, 'mask_detection_model.joblib')

import cv2
import numpy as np
import joblib  # Import joblib for loading the model
from tensorflow.keras.models import load_model

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained CNN model for mask detection
# Replace 'mask_detection_model.joblib' with the path to your saved model file
model = joblib.load('mask_detection_model.joblib')

# Define the labels for mask and no-mask
labels = {0: 'With Mask', 1: 'Without Mask'}

# Open a video capture stream or use an image
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a video file path

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to (150, 150) to match the model's input shape
        face_roi = cv2.resize(face_roi, (150, 150))

        # Convert the grayscale image to RGB by duplicating channels
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Normalize the pixel values
        face_roi_rgb = face_roi_rgb / 255.0

        # Expand the dimensions to create a batch of size 1
        face_roi_rgb = np.expand_dims(face_roi_rgb, axis=0)

        # Make predictions using the CNN model
        prediction = model.predict(face_roi_rgb)
        label_index = np.argmax(prediction)
        label = labels[label_index]
        confidence = prediction[0][label_index]

        # Draw a rectangle around the face and display the label and confidence
        color = (0, 255, 0) if label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the frame
    cv2.imshow('Mask Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# now we train a model for gender classification

import os
from keras.preprocessing.image import ImageDataGenerator
train_dir = os.path.join('gender/gender/', 'train')
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),class_mode='categorical')

validation_dir = os.path.join('gender/gender/', 'valid')
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150, 150),class_mode='categorical')

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))

from keras import optimizers
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

history = model.fit_generator(train_generator,epochs=10,validation_data=validation_generator)

import joblib
joblib.dump(model, 'gender_detection_model.joblib')

import cv2
import numpy as np
import joblib  # Import joblib for loading the model
from tensorflow.keras.models import load_model

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained CNN model for mask detection
# Replace 'mask_detection_model.joblib' with the path to your saved model file
model = joblib.load('gender_detection_model.joblib')

# Define the labels for mask and no-mask
labels = {0: 'Female', 1: 'Male'}

# Open a video capture stream or use an image
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a video file path

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to (150, 150) to match the model's input shape
        face_roi = cv2.resize(face_roi, (150, 150))

        # Convert the grayscale image to RGB by duplicating channels
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Normalize the pixel values
        face_roi_rgb = face_roi_rgb / 255.0

        # Expand the dimensions to create a batch of size 1
        face_roi_rgb = np.expand_dims(face_roi_rgb, axis=0)

        # Make predictions using the CNN model
        prediction = model.predict(face_roi_rgb)
        label_index = np.argmax(prediction)
        label = labels[label_index]
        confidence = prediction[0][label_index]

        # Draw a rectangle around the face and display the label and confidence
        color = (0, 255, 0) if label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the frame
    cv2.imshow('Gender Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# Now we combined both of the models and do prediction using webcam
import cv2
import numpy as np
import joblib  # Import joblib for loading the model
from tensorflow.keras.models import load_model

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained CNN models for gender and mask detection
gender_model = joblib.load('gender_detection_model.joblib')
mask_model = joblib.load('mask_detection_model.joblib')

# Define the labels for gender and mask
gender_labels = {0: 'Female', 1: 'Male'}
mask_labels = {0: 'With Mask', 1: 'Without Mask'}

# Open a video capture stream or use an image
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a video file path

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to (150, 150) to match the input shape of both models
        face_roi = cv2.resize(face_roi, (150, 150))

        # Convert the grayscale image to RGB by duplicating channels
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Normalize the pixel values
        face_roi_rgb = face_roi_rgb / 255.0

        # Expand the dimensions to create a batch of size 1
        face_roi_rgb = np.expand_dims(face_roi_rgb, axis=0)

        # Make predictions using the gender model
        gender_prediction = gender_model.predict(face_roi_rgb)
        gender_label_index = np.argmax(gender_prediction)
        gender_label = gender_labels[gender_label_index]

        # Make predictions using the mask model
        mask_prediction = mask_model.predict(face_roi_rgb)
        mask_label_index = np.argmax(mask_prediction)
        mask_label = mask_labels[mask_label_index]
        mask_confidence = mask_prediction[0][mask_label_index]

        # Draw a rectangle around the face and display the gender and mask labels
        color = (0, 255, 0) if mask_label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'Gender: {gender_label}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f'Mask: {mask_label} ({mask_confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the frame
    cv2.imshow('Combined Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# now we use two cameras and do predictions on both cameras
import cv2
import numpy as np
import joblib  # Import joblib for loading the model
from tensorflow.keras.models import load_model

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained CNN models for gender and mask detection
gender_model = joblib.load('gender_detection_model.joblib')
mask_model = joblib.load('mask_detection_model.joblib')

# Define the labels for gender and mask
gender_labels = {0: 'Female', 1: 'Male'}
mask_labels = {0: 'With Mask', 1: 'Without Mask'}

# Open two video capture streams for two different cameras
cap1 = cv2.VideoCapture(0)  # First camera (camera index 0)
cap2 = cv2.VideoCapture(http://192.168.100.1:8080/video)  # Second camera (camera index 1)

while True:
    # Read a frame from the first camera
    ret1, frame1 = cap1.read()

    # Read a frame from the second camera
    ret2, frame2 = cap2.read()

    # Convert the frames to grayscale for face detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frames from both cameras
    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces in the first camera
    for (x, y, w, h) in faces1:
        # Extract the face ROI
        face_roi = gray1[y:y + h, x:x + w]

        # Resize the face ROI to (150, 150) to match the input shape of both models
        face_roi = cv2.resize(face_roi, (150, 150))

        # Convert the grayscale image to RGB by duplicating channels
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Normalize the pixel values
        face_roi_rgb = face_roi_rgb / 255.0

        # Expand the dimensions to create a batch of size 1
        face_roi_rgb = np.expand_dims(face_roi_rgb, axis=0)

        # Make predictions using the gender model
        gender_prediction = gender_model.predict(face_roi_rgb)
        gender_label_index = np.argmax(gender_prediction)
        gender_label = gender_labels[gender_label_index]

        # Make predictions using the mask model
        mask_prediction = mask_model.predict(face_roi_rgb)
        mask_label_index = np.argmax(mask_prediction)
        mask_label = mask_labels[mask_label_index]
        mask_confidence = mask_prediction[0][mask_label_index]

        # Draw a rectangle around the face and display the gender and mask labels
        color = (0, 255, 0) if mask_label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame1, f'Gender: {gender_label}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame1, f'Mask: {mask_label} ({mask_confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Loop through the detected faces in the second camera
    for (x, y, w, h) in faces2:
        # Extract the face ROI
        face_roi = gray2[y:y + h, x:x + w]

        # Resize the face ROI to (150, 150) to match the input shape of both models
        face_roi = cv2.resize(face_roi, (150, 150))

        # Convert the grayscale image to RGB by duplicating channels
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Normalize the pixel values
        face_roi_rgb = face_roi_rgb / 255.0

        # Expand the dimensions to create a batch of size 1
        face_roi_rgb = np.expand_dims(face_roi_rgb, axis=0)

        # Make predictions using the gender model
        gender_prediction = gender_model.predict(face_roi_rgb)
        gender_label_index = np.argmax(gender_prediction)
        gender_label = gender_labels[gender_label_index]

        # Make predictions using the mask model
        mask_prediction = mask_model.predict(face_roi_rgb)
        mask_label_index = np.argmax(mask_prediction)
        mask_label = mask_labels[mask_label_index]
        mask_confidence = mask_prediction[0][mask_label_index]

        # Draw a rectangle around the face and display the gender and mask labels
        color = (0, 255, 0) if mask_label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame2, f'Gender: {gender_label}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame2, f'Mask: {mask_label} ({mask_confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display frames from both cameras
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()