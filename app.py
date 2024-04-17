import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the generator model
generator_model = load_model('generator_model.h5')

# Compile the model manually
generator_model.compile(optimizer='adam', loss='mean_squared_error')

# Set background color
st.set_page_config(
    page_title="Image Inpainting App",
    page_icon="🖼️",
    layout="wide",  # Wide layout to display larger images
    initial_sidebar_state="expanded",  # Expanded sidebar by default
    # Set background color to a mixture of #070e0e and #040708
    bg_color="linear-gradient(180deg, #070e0e, #040708)",
)


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
    input_img_resized = input_img.resize((128, 128))

    # Preprocess the image
    input_img_normalized = np.array(input_img_resized) / 255.0  # Normalize pixel values to [0, 1]

    # Generate masked image and missing parts
    masked_img, missing_parts = mask_center(
        np.expand_dims(input_img_normalized, axis=0))

    # Predict the missing parts
    predicted_missing_parts = generator_model.predict(masked_img)[0]

    # De-normalize the predicted missing parts
    predicted_missing_parts = np.clip(
        predicted_missing_parts * 255.0, 0, 255).astype(np.uint8)

    # Resize the predicted missing parts to match the size of the original image
    predicted_missing_parts_resized = Image.fromarray(predicted_missing_parts).resize((128, 128))

    # Combine the predicted missing parts with the original image
    reconstructed_img = np.where(
        masked_img[0] == 0, np.array(predicted_missing_parts_resized), np.array(input_img_resized))

    # Calculate PSNR
    psnr = calculate_psnr(np.array(input_img_resized), reconstructed_img)

    # Return both the reconstructed image, the masked image, and PSNR
    return reconstructed_img, masked_img[0], psnr


def calculate_psnr(original_img, reconstructed_img):
    # Convert images to float32
    original_img = original_img.astype(np.float32)
    reconstructed_img = reconstructed_img.astype(np.float32)
    # Calculate MSE
    mse = np.mean((original_img - reconstructed_img) ** 2)
    # Calculate maximum pixel value
    max_pixel = np.amax(original_img)
    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

# Streamlit app
st.title("Image Inpainting App")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    input_img = Image.open(uploaded_file)

    # Inpaint the image
    reconstructed_img, masked_img, psnr = reconstruct_image(input_img)

    st.image([input_img, Image.fromarray(reconstructed_img)], caption=[
             'Original Image','Reconstructed Image'], width=200)
    
    st.write(f"PSNR: {psnr:.2f}")
