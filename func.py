import typing
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

# Define constants
IMAGE_SIZE = 300
CLASS_NAMES = [
    "ba",
    "ca",
    "da",
    "dha",
    "ga",
    "ha",
    "ja",
    "ka",
    "la",
    "ma",
    "na",
    "nga",
    "nya",
    "pa",
    "ra",
    "sa",
    "ta",
    "tha",
    "wa",
    "ya",
]

# Load the model and get the input and output details
interpreter = tf.lite.Interpreter(model_path="model/f_4_normalized_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Define a helper function to format the time elapsed
def format_time(start_time: datetime) -> str:
    """
    Formats the time elapsed from a given start time to the current time.

    Args:
        start_time (datetime): The start time to measure elapsed time from.

    Returns:
        str: A formatted string representing the elapsed time.
    """
    thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
    tmin, tsec = divmod(temp_sec, 60)
    return "%i hours %i minutes and %s seconds." % (
        thour,
        tmin,
        round(tsec, 2),
    )


# Define a function to preprocess an image
@st.cache_data
def preprocess_image(image: Image) -> np.ndarray:
    """
    Resizes and normalizes an image.

    Args:
        image (Image): An image object.

    Returns:
        np.ndarray: A NumPy array representing the preprocessed image.
    """
    # Resize the image to the desired size
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    # Normalize the image
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


# Define a function to make predictions using the model
@st.cache_data
def predict_image(image: np.ndarray) -> dict:
    """
    Predicts the top 3 classes for an image using a pre-trained model.

    Args:
        image (np.ndarray): An image represented as a NumPy array.

    Returns:
        dict: A dictionary containing the top 3 predicted classes and their probabilities.
    """
    # Set the input tensor and invoke the model
    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    # Get the output tensor and sort the indices by descending probability
    output_data = interpreter.get_tensor(output_details[0]["index"])
    top_3_indices = np.argsort(output_data)[0, -3:][::-1]

    # Map the indices to the class names and probabilities
    predictions = {}
    for i in top_3_indices:
        predictions[CLASS_NAMES[i]] = output_data[0, i] * 100

    return predictions


# Define a function to handle the prediction request
@st.cache_data
def predict_func(data, from_numpy=False) -> str:
    """
    Handles the prediction request and returns a JSON response.

    Args:
        data: The input data, either a file object or a NumPy array.
        from_numpy (bool, optional): Whether the input data is a NumPy array or not. Defaults to False.

    Returns:
        str: A JSON string containing the predictions and the time taken.
    """
    # Start the timer
    start_time = datetime.now()

    # Convert the input data to an image object
    if not from_numpy:
        image = Image.open(data)
    else:
        image = Image.fromarray(data[:, :, :3])

    # Preprocess the image and make predictions
    image = preprocess_image(image)
    predictions = predict_image(image)

    # Format the response as a JSON string
    response = {
        "predictions": predictions,
        "time_taken": format_time(start_time),
    }
    response = json.dumps(response, indent=4)

    return response
