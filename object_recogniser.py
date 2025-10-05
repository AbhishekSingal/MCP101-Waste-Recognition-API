# object_recogniser.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the MobileNetV2 model pretrained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def recognise_image(img_array):
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)

    # Predict the image category using MobileNetV2
    preds = model.predict(x)

    # Decode the predictions into readable labels
    return decode_predictions(preds, top=3)[0]
