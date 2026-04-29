import streamlit as st
import numpy as np
import cv2

def mean_blur_manual(img, ksize=3):
    pad = ksize // 2
    if img.ndim == 3:
        img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j, c] = np.mean(img_pad[i:i+ksize, j:j+ksize, c])
    else:
        img_pad = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
        out = np.zeros_like(img)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.mean(img_pad[i:i+ksize, j:j+ksize])
    return out.astype(np.uint8)

def median_blur_manual(img, ksize=3):
    pad = ksize // 2
    if img.ndim == 3:
        img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j, c] = np.median(img_pad[i:i+ksize, j:j+ksize, c])
    else:
        img_pad = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
        out = np.zeros_like(img)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.median(img_pad[i:i+ksize, j:j+ksize])
    return out.astype(np.uint8)

def convolve2d(img, kernel):
    pad = kernel.shape[0] // 2
    img_pad = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img, dtype=float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img_pad[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return out

def sobel_edge_manual(gray):
    sobel_x = np.array([[ -1, 0, 1], [ -2, 0, 2], [ -1, 0, 1]])
    sobel_y = np.array([[ -1,-2,-1], [  0, 0, 0], [  1, 2, 1]])
    gx = convolve2d(gray, sobel_x)
    gy = convolve2d(gray, sobel_y)
    edge = np.sqrt(gx**2 + gy**2)
    return np.clip(edge, 0, 255).astype(np.uint8)

def hist_eq_manual(gray):
    hist, bins = np.histogram(gray.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
    cdf_final = np.ma.filled(cdf_masked,0).astype('uint8')
    equalized = cdf_final[gray]
    return equalized

def erode_manual(binary, ksize=3):
    pad = ksize // 2
    bin_pad = np.pad(binary, pad, mode='constant', constant_values=0)
    out = np.zeros_like(binary)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.min(bin_pad[i:i+ksize, j:j+ksize])
    return out

def dilate_manual(binary, ksize=3):
    pad = ksize // 2
    bin_pad = np.pad(binary, pad, mode='constant', constant_values=0)
    out = np.zeros_like(binary)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.max(bin_pad[i:i+ksize, j:j+ksize])
    return out

def threshold_manual(gray, thresh=128):
    binary = (gray > thresh) * 255
    return binary.astype(np.uint8)


def show_preprocess_image(base_dir=""):
    st.header("Image Preprocessing")
    st.info("Please preprocess an image here before continuing to the ASCII Art tab. After preprocessing, use the '→ Convert to ASCII' button below.")

    uploaded_file = st.file_uploader("Upload an image to preprocess", type=["jpg", "jpeg", "png"], key="pre_file")
    if not uploaded_file:
        st.stop()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, 1)  # Keep a copy BEFORE any processing!
    image = original.copy()

    # RESIZE
    resize_width = st.sidebar.slider("Resize Width", 64, 1024, image.shape[1])
    resize_height = st.sidebar.slider("Resize Height", 64, 1024, image.shape[0])
    image = cv2.resize(image, (resize_width, resize_height))

    # GRAYSCALE
    if st.sidebar.checkbox("Convert to Grayscale"):
        image = np.mean(image, axis=2).astype(np.uint8)  # Manual grayscale

    # BLUR OPTIONS
    blur_type = st.sidebar.selectbox("Blur Type", ["None", "Mean (Box)", "Median"])
    ksize = st.sidebar.slider("Blur Kernel Size", 3, 11, 3, step=2)
    if blur_type == "Mean (Box)":
        image = mean_blur_manual(image, ksize)
    elif blur_type == "Median":
        image = median_blur_manual(image, ksize)

    # HISTOGRAM EQUALISATION
    if st.sidebar.checkbox("Histogram Equalisation"):
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        image = hist_eq_manual(gray)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=2)

    # SOBEL EDGE DETECTION
    if st.sidebar.checkbox("Sobel Edge Detection"):
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        edge = sobel_edge_manual(gray)
        image = np.stack([edge]*3, axis=2)

    # THRESHOLDING
    if st.sidebar.checkbox("Thresholding"):
        thresh = st.sidebar.slider("Threshold", 0, 255, 128)
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        binary = threshold_manual(gray, thresh)
        image = np.stack([binary]*3, axis=2)

    # MORPHOLOGICAL OPERATIONS
    morph = st.sidebar.selectbox("Morphological Operation", ["None", "Erode", "Dilate"])
    morph_ksize = st.sidebar.slider("Morph Kernel Size", 3, 11, 3, step=2)
    if morph != "None":
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        if morph == "Erode":
            morphed = erode_manual(gray, morph_ksize)
        else:
            morphed = dilate_manual(gray, morph_ksize)
        image = np.stack([morphed]*3, axis=2)

    # SHOW THE ORIGINAL AND PREPROCESSED IMAGE SIDE BY SIDE
    def to_bgr(img):
        if img.ndim == 2:
            return np.stack([img]*3, axis=2)
        return img

    col1, col2 = st.columns(2)
    with col1:
        st.image(to_bgr(original), caption="Original Image", channels="BGR")
    with col2:
        st.image(to_bgr(image), caption="Preprocessed Image", channels="BGR")

    # SAVE to session_state for the next tab
    st.session_state["preprocessed_image"] = image

    # Button to go to ASCII tab
    if st.button("→ Convert to ASCII"):
        st.session_state.jump_to_ascii = True
        st.rerun()
