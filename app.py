import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the generator model
generator_model = load_model('saved_model/generator_model.h5')

# Compile the model manually
generator_model.compile(optimizer='adam', loss='mean_squared_error')


def mask_center(imgs, img_rows=128, img_cols=128, mask_height=32, mask_width=32, channels=3):
    center_y = img_rows // 2
    center_x = img_cols // 2
    half_height = mask_height // 2
    half_width = mask_width // 2

    y1 = np.full(imgs.shape[0], center_y - half_height)
    y2 = np.full(imgs.shape[0], center_y + half_height)
    x1 = np.full(imgs.shape[0], center_x - half_width)
    x2 = np.full(imgs.shape[0], center_x + half_width)

    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty(
        (imgs.shape[0], mask_height, mask_width, channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        missing_parts[i] = masked_img[y1[i]:y2[i], x1[i]:x2[i], :].copy()
        masked_img[y1[i]:y2[i], x1[i]:x2[i], :] = 0
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts


def reconstruct_image(input_img):
    # Resize to match model input size
    input_img_resized = cv2.resize(input_img, (128, 128))

    # Preprocess the image
    input_img_normalized = input_img_resized / \
        255.0  # Normalize pixel values to [0, 1]

    # Generate masked image and missing parts
    masked_img, missing_parts = mask_center(
        np.expand_dims(input_img_normalized, axis=0))

    # Predict the missing parts
    predicted_missing_parts = generator_model.predict(masked_img)[0]

    # De-normalize the predicted missing parts
    predicted_missing_parts = np.clip(
        predicted_missing_parts * 255.0, 0, 255).astype(np.uint8)

    # Resize the predicted missing parts to match the size of the original image
    predicted_missing_parts_resized = cv2.resize(
        predicted_missing_parts, (128, 128))

    # Combine the predicted missing parts with the original image
    reconstructed_img = np.where(
        masked_img[0] == 0, predicted_missing_parts_resized, input_img_resized)

    # Return both the reconstructed image and the masked image
    return reconstructed_img, masked_img[0]


# Streamlit app
st.title("Image Inpainting App")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    input_img = np.array(Image.open(uploaded_file))

    # Inpaint the image
    reconstructed_img, masked_img = reconstruct_image(input_img)

    # Display the original, masked, and reconstructed images
    st.image([input_img, masked_img, reconstructed_img], caption=[
             'Original Image', 'Masked Image', 'Reconstructed Image'], width=200)