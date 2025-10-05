import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

# Load the model once at the beginning
model = tf.keras.models.load_model("/Users/abhisheksingal/PycharmProjects/MCP100 Project/models/final_model_weights.hdf5")

def getPrediction(filename):
    img = load_img(filename, target_size=(180, 180))
    img = img_to_array(img)
    img = img / 255.0  # Normalizing the image
    img = np.expand_dims(img, axis=0)

    # Predict using the model
    predictions = model.predict(img)

    # Since it's binary classification, we expect predictions to have shape (1, 1) or (1, 2)
    if predictions.shape[1] == 1:  # This means your model has a single output (sigmoid activation)
        answer = (predictions[0][0] > 0.5).astype("int32")  # 0 for Organic, 1 for Recycle
        probability_results = predictions[0][0]  # Probability of the predicted class
    else:  # This means your model has two outputs (softmax activation)
        answer = np.argmax(predictions[0])  # Get the index of the class with the highest probability
        probability_results = predictions[0][answer]  # Probability of the predicted class

    # Assign the answer based on the predicted class
    if answer == 1:
        label = "Recycle"
    else:
        label = "Organic"

    return label, str(probability_results), filename

# Example usage
# result = getPrediction("/Users/abhisheksingal/PycharmProjects/MCP100 Project/resources/4.jpg")
# print(f"Prediction: {result[0]}, Probability: {result[1]}, File: {result[2]}")
