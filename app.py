import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Helmet Detection App :)")

# Load YOLO model
# model = YOLO("runs/detect/train73/weights/best.pt")
model = YOLO("runs/detect/train11/weights/best.pt")

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
  # Show original image
  st.image(uploaded_image , caption="Uploaded Image", use_container_width=True)

  # Read image and convert to numpy array
  image = Image.open(uploaded_image)
  image_np = np.array(image)

  # Run YOLO inference
  st.info("Running YOLO helmet detection...")
  results = model.predict(image_np , conf=0.4)

  # Draw results on image
  result_image = results[0].plot()
  st.image(result_image , caption="YOLO Detection Result", use_container_width=True)
  st.success("Detection completed!")

  # Extract detection results
  boxes = results[0].boxes
  class_ids = boxes.cls.cpu().numpy().astype(int)
  class_names = [model.names[i] for i in class_ids]

  # Count helmet
  helmet_count = class_names.count("helmet")
  st.write(f"Number of people with helmet detected: **{helmet_count}**")
  # without_helmet_count = class_names.count("w/o_helmet")
  # st.write(f"Number of people without helmet detected: **{without_helmet_count}**")
