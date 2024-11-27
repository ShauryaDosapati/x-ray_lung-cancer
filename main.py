import warnings
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

print(tf.__version__)
print(tf.keras.__version__)

# Image size for VGG16
IMAGE_SIZE = [224, 224]

# Paths to training and validation datasets
train_path = '/Users/shauryad/Developer/python/Datasets/LungCancerDetection/train'
valid_path = '/Users/shauryad/Developer/python/Datasets/LungCancerDetection/test'

# Load VGG16 model without the top layer and with specified input shape
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

# Define the number of classes based on folders
folders = glob(train_path + '/*')
num_classes = len(folders)

# Add layers to the model
x = Flatten()(vgg.output)
prediction = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Data augmentation for training and rescaling for test
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training and validation data
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=10,
    class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=10,
    class_mode='categorical'
)

# Display model summary to verify structure
model.summary()

# Train the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the trained model
model.save('chest_xray.h5')

# Load the saved model
model = load_model('chest_xray.h5')

# Test the model on a new image
img_path = '//Users/shauryad/Developer/python/Datasets/LungCancerDetection/test/Lungcancer/1f5b7158cd8d98f322fea84e9edbe84a.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)

# Predict the class probabilities
classes = model.predict(img_data)

# Find the predicted class with the highest probability
predicted_class = np.argmax(classes, axis=1)[0]

# Map the predicted class to label names (ensure the directory order matches)
labels = {v: k for k, v in training_set.class_indices.items()}  # Create a dictionary to map index to class name
result_label = labels[predicted_class]

# Display the result
if result_label == 'LUNG-CANCER':
    print("Person is Affected By Lung-Cancer")
else:
    print("Result is Normal or Other Class:", result_label)
