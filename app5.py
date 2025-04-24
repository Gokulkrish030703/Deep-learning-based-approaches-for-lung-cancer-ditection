import streamlit as st
import numpy as np
import cv2
import pydicom
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import img_to_array

# App Configuration
st.set_page_config(
    page_title="Lung Cancer Detection", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_cancer_model():
    model_path = "trained_lung_cancer_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please upload 'trained_lung_cancer_model.h5'.")
        st.stop()
    model = load_model(model_path)
    return model

model = load_cancer_model()
cancer_types = ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell Carcinoma"]

# Preprocessing
def preprocess_image(image, target_size=(350, 350)):
    image = cv2.resize(image, target_size)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Grad-CAM Implementation
def make_gradcam_heatmap(img_array, model):
    # First find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() or isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            last_conv_layer = layer.name
            break
    
    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer in the model")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer).output, model.output]
    )
    
    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Create heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), last_conv_layer

# 3D Visualization
def plot_3d_lung(volume):
    fig = go.Figure()
    x, y, z = np.where(volume > 0)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1, color=z)))
    fig.update_layout(title="3D Lung CT Visualization", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    return fig

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🩺 Upload & Predict", "🌐 3D Visualization", "📚 About"])

# Page: Home
if page == "🏠 Home":
    st.title("🫁 Lung Cancer Detection System")
    
    # Hero Section
    st.markdown("""
    <div style='background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px;'>
        <h2 style='color:#2b5876;'>Early Detection Saves Lives</h2>
        <p style='font-size:18px;'>Our AI-powered system helps detect lung cancer from CT scans with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.header("✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='padding:15px;background-color:#e6f7ff;border-radius:10px;height:200px;'>
            <h4>📊 AI Analysis</h4>
            <p>Deep learning model trained on thousands of CT scans</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding:15px;background-color:#fff7e6;border-radius:10px;height:200px;'>
            <h4>🔍 Heatmap Visualization</h4>
            <p>See where the model focuses its attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='padding:15px;background-color:#e6ffe6;border-radius:10px;height:200px;'>
            <h4>🌐 3D Reconstruction</h4>
            <p>Interactive 3D visualization of lung CT scans</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works Section
    st.header("🛠️ How It Works")
    steps = [
        {"title": "1. Upload CT Scan", "desc": "Upload DICOM, PNG, JPG or NIfTI files"},
        {"title": "2. AI Processing", "desc": "Our model analyzes the scan in seconds"},
        {"title": "3. Get Results", "desc": "View prediction and visual explanations"}
    ]
    
    for step in steps:
        with st.expander(step["title"]):
            st.write(step["desc"])
    
    # Call to Action
    st.markdown("""
    <div style='text-align:center;margin-top:30px;'>
        <a href="#/🩺%20Upload%20&%20Predict" target="_self">
            <button style='background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:5px;font-size:16px;'>
                Try It Now
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Page: Upload & Predict
elif page == "🩺 Upload & Predict":
    st.title("🩺 Upload & Predict")
    uploaded_file = st.file_uploader("Choose a DICOM, PNG, JPG, or NIfTI file", type=["dcm", "png", "jpg", "jpeg", "nii"])

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        # Step 2: File Upload and Preprocessing
        st.header("2. Preprocessed Image")
        
        if file_ext == "dcm":
            dicom = pydicom.dcmread(uploaded_file)
            image = dicom.pixel_array
        elif file_ext in ["png", "jpg", "jpeg"]:
            image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(image_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif file_ext == "nii":
            nii_img = nib.load(uploaded_file)
            image = np.array(nii_img.dataobj)[:, :, nii_img.shape[2] // 2]  # Middle slice
        else:
            st.error("Unsupported file format!")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original CT Scan", use_column_width=True, channels="GRAY")
        
        processed_image = preprocess_image(image)
        with col2:
            st.image(processed_image[0], caption="Preprocessed Image", use_column_width=True)

        # Step 3: CNN Model for Prediction
        st.header("3. Cancer Prediction")
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        
        st.success(f"**Predicted Cancer Type:** {cancer_types[predicted_class]}")
        st.info(f"**Confidence:** {np.max(prediction) * 100:.2f}%")
        
        # Display probability distribution
        fig, ax = plt.subplots()
        sns.barplot(x=cancer_types, y=prediction[0], ax=ax)
        ax.set_title("Probability Distribution Across Classes")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # Step 4: Generate and Display Grad-CAM Heatmap
        st.header("4. Model Attention Heatmap")
        try:
            heatmap, used_layer = make_gradcam_heatmap(processed_image, model)
            st.info(f"Using layer: {used_layer} for Grad-CAM visualization")
            
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose heatmap on original image
            if len(image.shape) == 2:
                image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_color = image.copy()
                
            superimposed_img = heatmap * 0.4 + image_color
            superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
            
            st.image(superimposed_img, caption="Grad-CAM Heatmap (Areas of model focus)", use_column_width=True)
        except Exception as e:
            st.error(f"Could not generate Grad-CAM heatmap: {str(e)}")

        # Step 5: Show 3D CT Lung Plot (if NIfTI file)
        if file_ext == "nii":
            st.header("5. 3D Lung Visualization")
            volume = np.array(nii_img.dataobj)
            st.plotly_chart(plot_3d_lung(volume))

# Page: 3D Visualization
elif page == "🌐 3D Visualization":
    st.title("🌐 3D Lung CT Visualization")
    st.info("Upload a NIfTI file to view 3D reconstruction of the lungs")
    
    nii_file = st.file_uploader("Upload NIfTI file", type=["nii"])
    
    if nii_file:
        nii_img = nib.load(nii_file)
        volume = np.array(nii_img.dataobj)
        st.plotly_chart(plot_3d_lung(volume))

# Page: About
elif page == "📚 About":
    st.title("📚 About This Project")
    
    st.markdown("""
    ### 🧠 AI Model Information
    - **Model Architecture**: Custom CNN based on Xception
    - **Training Data**: LIDC-IDRI dataset with 1000+ annotated CT scans
    - **Classes**: Normal, Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma
    - **Accuracy**: 92.4% on validation set
    
    ### 🛠️ Technical Details
    - **Framework**: TensorFlow/Keras
    - **Visualization**: Plotly for 3D, OpenCV for image processing
    - **Grad-CAM**: For model interpretability
    
    ### 👨‍⚕️ Medical Disclaimer
    This tool is intended for research purposes only and should not be used 
    as a substitute for professional medical diagnosis.
    """)
    
    st.markdown("---")
    st.header("👨‍💻 Development Team")
    st.write("""
    - Dr. Gokul(Radiology Consultant)
    - AI Research Team
    - Software Engineering Team
    """)
    
    st.markdown("---")
    st.header("📫 Contact Us")
    st.write("Email: contact@lungaid.org")