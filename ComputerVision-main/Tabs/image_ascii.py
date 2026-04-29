import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Standard character sets for intensity mapping
ASCII_SETS = {
    "Standard": "@%#*+=-:. ",
    "Minimalist": "#+-.",
    "Blocks": "█▓▒░ "
}

def convert_to_ascii(image, width_chars, palette):
    """Core logic: Maps 8-bit intensity to ASCII characters."""
    # 1. Aspect Ratio Correction: ASCII chars are usually taller than they are wide
    h, w = image.shape[:2]
    ratio = h / w
    height_chars = int(width_chars * ratio * 0.5)
    
    # 2. Downsample and Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width_chars, height_chars))
    
    # 3. Map intensity
    ascii_str = ""
    for row in resized:
        for pixel in row:
            # Map 0-255 to the index of our palette string
            index = int(pixel / 256 * len(palette))
            ascii_str += palette[index]
        ascii_str += "\n"
    return ascii_str

def show_image_ascii(base_dir=""):
    st.header("🖼️ Image to ASCII Art")
    
    if "preprocessed_image" not in st.session_state:
        st.error("Upload an image in the Preprocess tab first!")
        return

    img = st.session_state["preprocessed_image"]
    
    # UI Controls for Resolution and Palette
    char_width = st.sidebar.slider("Resolution (Chars Wide)", 20, 200, 100)
    palette_name = st.sidebar.selectbox("Character Set", list(ASCII_SETS.keys()))
    
    ascii_output = convert_to_ascii(img, char_width, ASCII_SETS[palette_name])
    
    # Feature: Automatic Copying
    st.subheader("ASCII Result (Copy-friendly)")
    st.code(ascii_output) # Built-in copy button
    
    # Feature: Text Download
    st.download_button("📩 Download as Text", ascii_output, "art.txt")