import streamlit as st
from pathlib import Path

# Application configuration 
st.set_page_config(page_title="University Vision Project", layout="wide")

PAGES = ["Preprocess Image", "Image to ASCII Art", "Webcam to ASCII Art"]

if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]

st.sidebar.title("Navigation")
st.session_state["page"] = st.sidebar.radio("Go to", PAGES)

page = st.session_state["page"]
base_dir = Path(__file__).resolve().parent

# Dynamic Routing 
if page == "Preprocess Image":
    from Tabs.preprocess_image import show_preprocess_image
    show_preprocess_image(base_dir)
elif page == "Image to ASCII Art":
    from Tabs.image_ascii import show_image_ascii
    show_image_ascii(base_dir)
elif page == "Webcam to ASCII Art":
    from Tabs.webcam_ascii import show_webcam_ascii
    show_webcam_ascii(base_dir)