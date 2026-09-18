import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm
import datetime

# Initialize prediction history in Streamlit session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Grad-CAM Functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.6):
    img = np.array(img.convert("RGB"))
    
    # Rescale heatmap to a 0-255 uint8 array
    heatmap = np.uint8(255 * heatmap)
    
    # Use matplotlib jet colormap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Resize heatmap to match original image size
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on the original image with alpha
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = Image.fromarray(superimposed_img)
    return superimposed_img

# Page Configuration
st.set_page_config(
    page_title="VisionAI - General Image Recognition", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache and Load Model
@st.cache_resource
def load_recognition_model():
    return MobileNetV2(weights='imagenet')

with st.spinner("⏳ Loading MobileNetV2 AI Engine... Please wait."):
    model = load_recognition_model()

# Sidebar Information Panel
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("About VisionAI")
    st.info(
        "This application uses **MobileNetV2** pre-trained on the **ImageNet** dataset. "
        "It can instantly recognize over 1,000 everyday object categories, animals, food, vehicles, and insects!"
    )
    st.markdown("---")
    st.markdown("🛠️ **Built with:** TensorFlow & Streamlit")

# Main Content Area
st.markdown('<p class="main-title">🌟 AI-Based General Image Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload any image to analyze its contents and view AI explanation heatmaps instantly.</p>', unsafe_allow_html=True)

# Layout Split: Left for Upload & Image, Right for Results
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📂 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Test Image", use_container_width=True)

with col2:
    st.markdown("### 📊 Recognition Results & Explainability")
    
    if image is not None:
        if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔍 Extracting features and predicting..."):
                # Resize and preprocess image to 224x224 for MobileNetV2
                img_resized = image.resize((224, 224))
                x = img_to_array(img_resized)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                # Predict
                preds = model.predict(x)
                decoded_preds = decode_predictions(preds, top=3)[0]
                
                # Show Results
                st.success("Analysis Complete Successfully!")
                st.markdown("---")
                
                # Display Top 3 Predictions
                for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                    formatted_label = label.replace('_', ' ').title()
                    confidence = score * 100
                    
                    if i == 0:
                        top_label = formatted_label
                        top_conf = confidence
                        st.metric(label=f"🥇 Top Match: {formatted_label}", value=f"{confidence:.2f}%")
                        st.progress(float(score))
                    elif i == 1:
                        st.metric(label=f"🥈 2nd Choice: {formatted_label}", value=f"{confidence:.2f}%")
                    else:
                        st.metric(label=f"🥉 3rd Choice: {formatted_label}", value=f"{confidence:.2f}%")
                
                # Log prediction to session history (Fixed indentation scope)
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                scan_record = {
                    "Time": current_time,
                    "Image Name": uploaded_file.name,
                    "Top Prediction": top_label,
                    "Confidence": f"{top_conf:.2f}%"
                }
                st.session_state.history.insert(0, scan_record)
                
                # Generate and Display Grad-CAM Heatmap
                st.markdown("---")
                st.subheader("🔥 AI Attention Heatmap (Grad-CAM)")
                last_conv_layer_name = "Conv_1"
                pred_index = np.argmax(preds[0])
                heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name, pred_index)
                gradcam_image = display_gradcam(image, heatmap)
                st.image(gradcam_image, caption="Highlighted areas show what the AI focused on", use_container_width=True)
    else:
        st.info("👈 Please upload an image using the left panel to begin the analysis.")

# --- Prediction History & Analytics Section ---
st.markdown("---")
st.subheader("📋 Session Prediction History & Analytics")

if len(st.session_state.history) > 0:
    col_hist1, col_hist2 = st.columns([2, 1])
    
    with col_hist1:
        st.markdown("**Recent Scans Log:**")
        st.dataframe(st.session_state.history, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
            
    with col_hist2:
        st.metric(label="Total Images Scanned", value=len(st.session_state.history))
        top_preds = [rec["Top Prediction"] for rec in st.session_state.history]
        most_common = max(set(top_preds), key=top_preds.count) if top_preds else "N/A"
        st.metric(label="Most Frequent Category", value=most_common)
else:
    st.info("No prediction history yet. Upload and analyze an image to start logging scans!")