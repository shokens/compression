import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import restoration, exposure, morphology, feature, img_as_ubyte
from sklearn.cluster import KMeans
import io
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import pytesseract
import matplotlib

matplotlib.use('Agg')

# ======================= Core Application Setup =======================
st.set_page_config(page_title="Advanced Image Processing Platform By System Engineering MSC Student", layout="wide")
st.title("🧠 ML-Powered Digital Image Processing & Compression Web Platform By System Engineering MSC Student at NARSDA")

# ======================= Processing Stage Controls =======================
st.sidebar.title("🔧 Processing Pipeline Configuration")

# Main processing stages
processing_stages = {
    "Acquisition": st.sidebar.checkbox("1. Image Acquisition"),
    "Conversion": st.sidebar.checkbox("2. Image Import/Conversion"),
    "Preprocessing": st.sidebar.checkbox("3. Image Preprocessing"),
    "Restoration": st.sidebar.checkbox("4. Image Restoration"),
    "Enhancement": st.sidebar.checkbox("5. Image Enhancement"),
    "Segmentation": st.sidebar.checkbox("6. Image Segmentation"),
    # "Feature": st.sidebar.checkbox("7. Feature Extraction"),
    # "Morphological": st.sidebar.checkbox("8. Morphological Processing"),
    # "Analysis": st.sidebar.checkbox("9. Image Analysis"),
    "Compression": st.sidebar.checkbox("7. Compression"),
    # "Synthesis": st.sidebar.checkbox("11. Image Synthesis")
}

# ======================= File Upload Section =======================
uploaded_file = st.file_uploader("📤 Upload Image/PDF", type=["jpg", "png", "jpeg", "pdf", "tiff", "bmp"])

# ======================= Processing Functions =======================
def handle_acquisition():
    st.header("1. Image Acquisition")
    col1, col2 = st.columns(2)
    with col1:
        source_type = st.radio("Input Source:", 
                            ["File Upload", "Webcam Simulation"])
        
    with col2:
        if source_type == "Webcam Simulation":
            st.warning("Webcam access requires browser permission")
            st.image(np.random.rand(300,400,3), caption="Live Webcam Feed")
        elif source_type == "Medical Imaging":
            st.image("mri_sample.jpg", caption="MRI Scan Demo")
        elif source_type == "Satellite":
            st.image("satellite_sample.jpg", caption="Satellite Imagery Demo")

def handle_conversion(img):
    st.header("2. Image Conversion")
    col1, col2 = st.columns(2)
    
    with col1:
        color_space = st.selectbox("Color Space:", ["RGB", "GRAY", "HSV", "LAB", "YCbCr"])
        bit_depth = st.select_slider("Bit Depth:", [8, 16, 32])
        resize_factor = st.slider("Resize Factor", 0.1, 2.0, 1.0)
    
    with col2:
        # Color space conversion
        if color_space != "RGB":
            conversion_code = getattr(cv2, f"COLOR_RGB2{color_space}")
            converted = cv2.cvtColor(img, conversion_code)
        else:
            converted = img.copy()
        
        # Bit depth conversion
        if bit_depth != 8:
            converted = (converted * (2**bit_depth / 256)).astype(f'uint{bit_depth}')
        
        # Resizing
        h, w = converted.shape[:2]
        converted = cv2.resize(converted, (int(w*resize_factor), int(h*resize_factor)))
        
        st.image(converted, caption=f"Converted Image ({color_space}, {bit_depth}-bit)")
    return converted

def handle_preprocessing(img):
    st.header("3. Image Preprocessing")
    options = st.multiselect("Select Operations:", 
                           ["Noise Reduction", "Contrast Adjustment", "Filtering",
                            "Thresholding", "Geometric Transform"])
    
    processed = img.copy()
    
    if "Noise Reduction" in options:
        kernel = st.slider("Gaussian Kernel Size", 3, 15, 5, step=2)
        processed = cv2.GaussianBlur(processed, (kernel, kernel), 0)
    
    if "Contrast Adjustment" in options:
        alpha = st.slider("Contrast Factor", 0.0, 3.0, 1.0)
        processed = cv2.convertScaleAbs(processed, alpha=alpha)
    
    if "Thresholding" in options:
        thresh = st.slider("Threshold Value", 0, 255, 127)
        _, processed = cv2.threshold(processed, thresh, 255, cv2.THRESH_BINARY)
    
    return processed

def handle_restoration(img):
    st.header("4. Image Restoration")
    method = st.selectbox("Restoration Method:", 
                        ["Non-local Means", "Wavelet Denoising", "Inpainting"])
    
    if method == "Non-local Means":
        denoised = restoration.denoise_nl_means(
            img, fast_mode=True, patch_size=5, patch_distance=3, h=0.1, channel_axis=-1
        )
        processed = img_as_ubyte(denoised)
    elif method == "Wavelet Denoising":
        processed = restoration.denoise_wavelet(img, channel_axis=-1)
    else:
        st.warning("Inpainting not implemented in this version")
        processed = img.copy()
    
    st.image(processed, caption=f"{method} Result")
    return processed

def handle_enhancement(img):
    st.header("5. Image Enhancement")
    option = st.radio("Enhancement Type:", 
                    ["Histogram Equalization", "Sharpening", "Contrast Stretching"])
    
    if option == "Histogram Equalization":
        if len(img.shape) == 3:
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            processed = cv2.equalizeHist(img)
    elif option == "Sharpening":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(img, -1, kernel)
    else:
        p2, p98 = np.percentile(img, (2, 98))
        processed = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    st.image(processed, caption=option)
    return processed

def handle_segmentation(img):
    st.header("6. Image Segmentation")
    method = st.selectbox("Segmentation Method:", 
                        [ "Edge Detection", "Thresholding"])
    
    if method == "K-Means Clustering":
        k = st.slider("Number of Clusters", 2, 8, 4)
        flat_img = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(flat_img)
        processed = kmeans.cluster_centers_[kmeans.labels_].reshape(img.shape)
    elif method == "Edge Detection":
        processed = cv2.Canny(img, 100, 200)
    else:
        _, processed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    st.image(processed, caption=method)
    return processed

def handle_compression(img):
    st.header("10. Compression")
    col1, col2 = st.columns(2)
    
    with col1:
        comp_type = st.radio("Compression Type:", ["JPEG", "PNG", "WebP"])
        quality = st.slider("Quality", 1, 100, 50)
    
    with col2:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode(f'.{comp_type.lower()}', img, encode_param)
        processed = cv2.imdecode(enc_img, 1)
        
        st.image(processed, caption=f"{comp_type} Compression (Quality: {quality}%)")
        buf = io.BytesIO()
        Image.fromarray(processed).save(buf, format=comp_type)
        st.download_button(f"Download {comp_type}", buf.getvalue(), f"compressed.{comp_type}")
    
    return processed

# ======================= Main Processing Pipeline =======================
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.header("PDF Processing")
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        if st.button("Compress PDF"):
            output = fitz.open()
            for page in doc:
                pix = page.get_pixmap(dpi=100)
                img_bytes = pix.tobytes("jpeg", quality=50)
                img_pdf = fitz.open("pdf", img_bytes)
                output.insert_pdf(img_pdf)
            
            buf = io.BytesIO()
            output.save(buf)
            st.download_button("Download Compressed PDF", buf.getvalue(), "compressed.pdf")
    else:
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)
        
        if processing_stages["Acquisition"]:
            handle_acquisition()
        
        if processing_stages["Conversion"]:
            img_np = handle_conversion(img_np)
        
        if processing_stages["Preprocessing"]:
            img_np = handle_preprocessing(img_np)
        
        if processing_stages["Restoration"]:
            img_np = handle_restoration(img_np)
        
        if processing_stages["Enhancement"]:
            img_np = handle_enhancement(img_np)
        
        if processing_stages["Segmentation"]:
            img_np = handle_segmentation(img_np)
        
        if processing_stages["Compression"]:
            img_np = handle_compression(img_np)
        
        st.header("Final Output")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image")
        with col2:
            st.image(img_np, caption="Processed Image")

# ======================= System Controls =======================
st.sidebar.header("⚙️ System Settings")
st.sidebar.checkbox("Enable GPU Acceleration")
st.sidebar.checkbox("Use Deep Learning Models")
st.sidebar.selectbox("Log Level:", ["Debug", "Info", "Warning", "Error"])

st.sidebar.header("📤 Export Options")
st.sidebar.button("Save Configuration")
st.sidebar.button("Export Report")