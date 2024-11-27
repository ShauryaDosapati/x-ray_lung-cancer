import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)

# Image size for VGG16
IMAGE_SIZE = [224, 224]

# Paths
train_path = '/Users/shauryad/Developer/python/Datasets/LungCancerDetection/train'
img_path = '/Users/shauryad/Developer/python/Datasets/LungCancerDetection/test/Lungcancer/2a64637b8ad5881733ef45dcbbdaf127.jpg'

# Load the model
model = load_model('chest_xray.h5')

# Check class mappings
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=10,
    class_mode='categorical'
)
class_labels = training_set.class_indices
labels = {v: k for k, v in class_labels.items()}  # Map indices to class names
print("Class Labels:", labels)

# Load and preprocess the image
img = image.load_img(img_path, target_size=IMAGE_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0  # Normalize

# Predict the class
classes = model.predict(x)
predicted_class = np.argmax(classes, axis=1)[0]
result_label = labels[predicted_class]

# Display the result
if "lung" in result_label.lower():
    print("Person is Affected By Lung-Cancer")
else:
    print("Result is Normal or Other Class:", result_label)
