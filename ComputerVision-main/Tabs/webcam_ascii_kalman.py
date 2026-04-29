import cv2
import os
import time
import threading
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from ultralytics import YOLO
from pathlib import Path
import mediapipe as mp


# Simple 2D Kalman Filter for landmark smoothing
class Kalman2D:
    def __init__(self, q=0.01, r=1.0):
        self.q = q
        self.r = r
        self.p = np.eye(2)
        self.x = np.zeros((2,))
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = np.array(measurement)
            self.initialized = True
        self.p += self.q * np.eye(2)
        k = self.p / (self.p + self.r)
        self.x = self.x + k @ (np.array(measurement) - self.x)
        self.p = (np.eye(2) - k) @ self.p
        return tuple(self.x.astype(int))

# Font detection (cross-platform)
possible_fonts = [
    "C:/Windows/Fonts/consola.ttf", 
    "C:/Windows/Fonts/lucon.ttf",    
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  
    "/System/Library/Fonts/Menlo.ttc"
]
FONT_PATH = next((f for f in possible_fonts if Path(f).exists()), None)
if FONT_PATH is None:
    raise FileNotFoundError("No suitable monospace font found.")

# ASCII gradients
ASCII_GRADIENTS = {
    "Minimalist": ".:-=+*#%@",
    "Medium set": "@%#*+=-:. ",
    "Numerical": "0123456789"
}

# Models and trackers
model = YOLO("yolov8n.pt")
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

# Thread-safe webcam variables
frame_lock = threading.Lock()
frame_buffer = None
cap_thread = None
cap_running = False

# Landmark Kalman filters
filters = {}

# Image enhancements
def apply_enhancements(img_pil, brightness, contrast, saturation):
    img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    img_pil = ImageEnhance.Color(img_pil).enhance(saturation)
    return img_pil

# Convert image to ASCII
def image_to_ascii_image(img, width, height, font_path, font_size, gradient, invert):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = 255 - gray
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    cols = max(1, width // char_width)
    rows = max(1, int((height // char_height) * (char_height / char_width)))
    resized = cv2.resize(gray, (cols, rows))
    ascii_art = [
        "".join([gradient[int(p / 255 * (len(gradient) - 1))] for p in row])
        for row in resized
    ]
    img_ascii = Image.new("RGB", (cols * char_width, rows * char_height), (255, 255, 255))
    draw = ImageDraw.Draw(img_ascii)
    for i, line in enumerate(ascii_art):
        draw.text((0, i * char_height), line, font=font, fill=(0, 0, 0))
    return cv2.resize(np.array(img_ascii), (width, height))

# Webcam reader loop
def webcam_reader():
    global cap_running, frame_buffer
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while cap_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                frame_buffer = frame.copy()
        time.sleep(0.03)
    cap.release()

# Streamlit interface
def show_webcam_ascii(base_dir=""):
    global cap_thread, cap_running
    st.header("Live Webcam to ASCII Art")

    if "ascii_stream" not in st.session_state:
        st.session_state.ascii_stream = False

    # Sidebar settings
    st.sidebar.header("Live Settings")
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    saturation = st.sidebar.slider("Saturation", 0.0, 2.0, 1.0)
    grayscale = st.sidebar.checkbox("Grayscale", False)
    invert = st.sidebar.checkbox("Invert Colors", False)
    font_size = st.sidebar.slider("Font Size", 6, 20, 8)
    gradient = st.sidebar.selectbox("Gradient", list(ASCII_GRADIENTS.keys()))
    fps_skip = st.sidebar.slider("Frame Skip", 1, 4, 2)
    ascii_mode = st.radio("ASCII Mode", ("Full Frame", "YOLO Boxes with ASCII"), key="ascii_mode")
    show_landmarks = st.checkbox("Show Face Landmarks", True)
    use_kalman = st.checkbox("Apply Kalman Filter", True)

    placeholder = st.empty()
    ascii_frames = []

    if not st.session_state.ascii_stream:
        if st.button("▶ Start Webcam Stream"):
            st.session_state.ascii_stream = True
            cap_running = True
            cap_thread = threading.Thread(target=webcam_reader)
            cap_thread.start()
            st.rerun()
    else:
        if st.button("⏹ Stop Webcam Stream"):
            cap_running = False
            st.session_state.ascii_stream = False
            if cap_thread:
                cap_thread.join()

            # Save GIF after stopping
            if ascii_frames:
                gif_path = "ascii_output.gif"
                ascii_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=ascii_frames[1:],
                    duration=100,
                    loop=0,
                    optimize=True,
                )
                with open(gif_path, "rb") as f:
                    st.download_button("⬇ Download ASCII Art as GIF", f.read(), "ascii_output.gif", "image/gif")

            st.rerun()

    # Main streaming loop
    if st.session_state.ascii_stream:
        i = 0
        while st.session_state.ascii_stream and cap_running:
            if i % fps_skip != 0:
                i += 1
                continue
            i += 1

            with frame_lock:
                if frame_buffer is None:
                    continue
                frame = frame_buffer.copy()
            modified = frame.copy()

            # ASCII logic
            if ascii_mode == "YOLO Boxes with ASCII":
                results = model(modified)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes is not None else []
                for (x1, y1, x2, y2) in boxes:
                    roi = frame[y1:y2, x1:x2]
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    roi_pil = apply_enhancements(roi_pil, brightness, contrast, saturation)
                    if grayscale:
                        roi_pil = roi_pil.convert("L").convert("RGB")
                    roi = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
                    ascii_img = image_to_ascii_image(roi, x2 - x1, y2 - y1, FONT_PATH, font_size, ASCII_GRADIENTS[gradient], invert)
                    modified[y1:y2, x1:x2] = ascii_img
            else:
                roi = frame.copy()
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_pil = apply_enhancements(roi_pil, brightness, contrast, saturation)
                if grayscale:
                    roi_pil = roi_pil.convert("L").convert("RGB")
                roi = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
                ascii_img = image_to_ascii_image(roi, roi.shape[1], roi.shape[0], FONT_PATH, font_size, ASCII_GRADIENTS[gradient], invert)
                modified = ascii_img

            # Landmark overlay
            if show_landmarks:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extended list of important landmarks
                        landmark_indices = [1, 10, 33, 61, 65, 152, 199, 234, 263, 291, 295, 454]
                        for idx in landmark_indices:
                            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                            if use_kalman:
                                if idx not in filters:
                                    filters[idx] = Kalman2D()
                                x, y = filters[idx].update((x, y))
                            cv2.putText(modified, "@", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            placeholder.image(modified, channels="BGR")
            # Save this frame for GIF
            ascii_frames.append(Image.fromarray(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)))
            time.sleep(0.03)
