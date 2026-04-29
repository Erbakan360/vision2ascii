import cv2
import streamlit as st
import numpy as np
from PIL import Image

def show_webcam_ascii(base_dir=""):
    st.header("🎥 Live Camera ASCII")
    
    # Access controls in the sidebar
    res = st.sidebar.slider("Live Resolution", 40, 150, 80)
    run = st.checkbox("Toggle Camera")
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    # Using st.empty to update the frame in place
    frame_placeholder = st.empty()

    while run:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to access camera.")
            break
            
        # Simplified ASCII conversion for real-time speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        ratio = h / w
        small = cv2.resize(gray, (res, int(res * ratio * 0.5)))
        
        # Simple intensity mapping
        chars = "@%#*+=-:. "
        ascii_frame = ""
        for row in small:
            ascii_frame += "".join([chars[int(p/256*len(chars))] for p in row]) + "\n"
        
        # Rendering
        frame_placeholder.code(ascii_frame)
        
    cam.release()