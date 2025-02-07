import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import io
from PIL import Image
import os
import gdown

# Set page configuration for full responsiveness
st.set_page_config(page_title="FloraFly 🍃🦋", layout="wide")

# Google Drive file ID
file_id = "161Wwj3OAtVXHRCsztE2VxWttDDAKWkI1"
model_path = "vgg16_fold_1.keras"

# Download the model if not exists
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ["Hesperiidae", "Lycaenidae", "Papilionidae", "Pieridae", "Nymphalidae"]

# Function to generate Grad-CAM heatmap
vgg16_for_gradcam = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def get_gradcam_heatmap(img_array, class_idx, last_conv_layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model(
        inputs=[vgg16_for_gradcam.input],
        outputs=[
            vgg16_for_gradcam.get_layer(last_conv_layer_name).output,
            vgg16_for_gradcam.output
        ]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay Grad-CAM on the image
def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

st.title("FloraFly 🍃🦋")
tabs = st.tabs([
    "🏠 Home", "📤 Upload Image", "🔍 Classification", "📊 Visualizations", "🦋 Gallery & History", "ℹ️ About & Contact"
])

# Tab 1: Home
with tabs[0]:
    st.header("🌿 Welcome to FloraFly! 🍃🦋")
    st.subheader("Discover. Identify. Conserve.")

    st.write(
    "🌿🦋 **Nature's little wonders are just a click away!** "
    "Welcome to **FloraFly**, an AI-powered butterfly classification tool designed to assist researchers, conservationists, and nature enthusiasts. "
    "With cutting-edge deep learning technology, **FloraFly** makes it easy to **identify butterfly families effortlessly**, helping you explore and understand butterfly biodiversity like never before!"
    )

    # Create an engaging layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(r"alfred-schrock-shoE2y51uX0-unsplash.jpg", caption="Explore Butterfly Species", use_container_width=True)

    with col2:
        st.write("### 🦋 How It Works?")
        st.write("Our AI-powered butterfly identification tool makes classification effortless and insightful. Here's how you can use it:")

        st.write("✅ **Upload or Capture** – Snap a real-time photo using your device's camera or upload an image from your gallery. Supported formats include JPG and PNG.")

        st.write("✅ **AI-Powered Recognition** – Leveraging state-of-the-art deep learning models, the system swiftly analyzes your image and predicts the butterfly’s family with high accuracy.")

        st.write("✅ **Interactive Visuals** – Gain deeper insights into AI-driven decision-making with Grad-CAM heatmaps. These visualizations highlight the most influential regions of the butterfly that contributed to the classification.")

        st.write("✅ **Seamless Experience** – Enjoy a user-friendly interface with real-time processing, ensuring an intuitive and engaging way to explore butterfly biodiversity.")

    # Highlight the purpose
    st.write("### 🌍 Why Use FloraFly?")
    st.write("🔬 **Precision-Driven AI** – Harnessing cutting-edge deep learning models, FloraFly ensures accurate and reliable butterfly family identification.")
    st.write("📸 **User-Friendly & Interactive** – Simply upload an image or capture one in real-time to receive instant results with intuitive visualizations.")
    st.write("🦋 **Empowering Conservation & Research** – A valuable tool for researchers, conservationists, and nature enthusiasts to study and protect butterfly biodiversity.")

    # CTA section
    st.markdown("### **🦋 Ready to Explore?**")
    st.write("Upload an image and let **FloraFly** unveil the magic of butterfly identification! ✨")

    # Reset the stored image if not available
    if "image_array" not in st.session_state:
        st.session_state.image_array = None


# Tab 2: Upload Image
with tabs[1]:
    st.header("📤 Upload an Image")
    st.subheader("🌿 Capture the beauty of butterflies & let **FloraFly** do the magic! 🦋✨")

    st.write(
        "Simply **upload an image** or **take a photo**, and our AI model will classify it into one of the butterfly families. "
        "Ensure the image is **clear** and **focused** for the best results."
    )

    # Add spacing and interactive layout
    st.markdown("---")  # Horizontal divider for a cleaner look

    choice = st.radio("📷 How would you like to provide an image?", ("Upload from device", "Take a photo"))

    image_source = None
    col1, col2 = st.columns(2)  # Split layout for better design

    with col1:
        if choice == "Upload from device":
            uploaded_file = st.file_uploader("📁 **Choose an image:**", type=["jpg", "png"])
            if uploaded_file:
                image_source = uploaded_file

    with col2:
        if choice == "Take a photo":
            captured_image = st.camera_input("📸 **Take a picture:**")
            if captured_image:
                image_source = captured_image

    # Display image and handle errors
    if image_source:
        file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)

        if file_bytes.size == 0:
            st.error("❌ Error: The uploaded file is empty. Please try again with a valid image.")
        else:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("⚠️ Error: OpenCV failed to decode the image. Ensure the file is a valid image format.")
            else:
                # Resize the image to a smaller size for display
                display_img = cv2.resize(img, (224, 224))  # Adjusted to a reasonable fixed size
                
                # Convert color format for Streamlit display
                display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

                # Display the resized uploaded image without stretching
                st.image(display_img_rgb, caption="🖼️ **Uploaded Image**", width=224)

                st.success("✅ **Image loaded successfully!** Ready for classification. 🚀")
                
                # Save the original image (not resized) for classification
                st.session_state.image_array = display_img_rgb


    # Add a final encouragement section
    st.markdown("---")
    st.write("🌟 **Tip:** For best results, use high-quality images with a clear butterfly view with image size 224x224.")



# Tab 3: Classification
with tabs[2]:
    st.header("🔍 Classification Results")
    st.subheader("🌿 AI-Powered Butterfly Identification 🦋")

    if st.session_state.image_array is None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])  # Adjust column width ratios

        with col1:
            # Display an illustration for guidance
            st.image(r"krzysztof-niewolny-T3EzlprIYDk-unsplash.jpg", caption="🦋 Upload an image to classify!", use_container_width=True)

        with col2:
            # Add guidance for the user
            st.markdown("""
            ### 🛠 How It Works?
            1. **Navigate to the 'Upload Image' tab** – Capture a photo or upload an image of a butterfly.
            2. **AI-Powered Classification** – Our deep learning model quickly analyzes the image and predicts the butterfly family.
            3. **Detailed Insights & Visualizations** – View confidence scores, AI-driven Grad-CAM heatmaps, and classification metrics to understand the model's decision-making.

            🔍 **Upload an image now and let AI reveal the butterfly's identity!** 🦋
            """)

        
        # Display a message when no image is uploaded
        st.info("📸 **No image uploaded yet.** Please upload an image in the **Upload Image** tab to see classification results.")

    else:
        # Resize uploaded image for display only
        display_img = cv2.resize(st.session_state.image_array, (224, 224))  # Display size adjustment

        # Display resized image
        st.image(display_img, caption="📸 **Uploaded Image**", width=224)

        # Preprocess the image for prediction (original resolution)
        img_resized = cv2.resize(st.session_state.image_array, (224, 224))
        img_array = np.expand_dims(img_resized.astype("float32") / 255, axis=0)

        # Predict the butterfly family
        preds = model.predict(img_array)[0]
        sorted_indices = np.argsort(preds)[::-1]

        predicted_class = sorted_indices[0]
        predicted_label = class_labels[predicted_class]
        confidence = preds[predicted_class]

        # Display main prediction in an elegant way
        st.markdown(f"""
            <div style="background-color: #6d6d6d; padding: 15px; border-radius: 10px; text-align: center;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #2b2313;">✨ Predicted Family: <b>{predicted_label}</b> 🦋</h3>
                <h4>Confidence Score: <b>{confidence:.2f}</b> 🔥</h4>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Show confidence levels for all predictions with progress bars
        st.write("### 🏆 Confidence Levels for All Families:")
        for idx in sorted_indices:
            st.write(f"**{class_labels[idx]}** ({preds[idx]:.2f})")
            st.progress(float(preds[idx]))  # Dynamic progress bar

        # Add a call to action
        st.markdown("---")
        st.write("🎯 **Want to explore more?** Check the Grad-CAM visualization to see what influenced the model’s decision!")

# Tab 4: Grad-CAM Visualization
with tabs[3]:
    st.header("🎨 Grad-CAM Visualizations")

    if st.session_state.image_array is None:
        # Display placeholder when no image is uploaded
        
        col1, col2 = st.columns([1, 1])
        # Display a placeholder butterfly image
        with col1:
            st.image(r"yuichi-kageyama-4ByFHyNdoD4-unsplash.jpg", caption="🦋 Upload an image to generate a heatmap!", use_container_width=True)
        
        # Explain what Grad-CAM is in an interactive way
        with col2:
            st.markdown("""
            ### 🔎 What is Grad-CAM?
            Grad-CAM (**Gradient-weighted Class Activation Mapping**) is an **AI-powered visualization technique** that helps you understand how the model makes predictions. It highlights the most influential regions of an image that contributed to the classification.

            🔹 **How It Works:**  
            - The model analyzes the image and assigns importance to different regions.  
            - Areas with **intense red coloration** indicate the most crucial features used for classification. 🔥  
            - This helps researchers and users interpret AI decisions and gain insights into butterfly characteristics.

            👉 **Upload an image** to generate a Grad-CAM heatmap and explore how AI sees the world of butterflies! 🦋
            """)

        st.info("📸 **No image uploaded yet.** Please upload an image in the **Upload Image** tab to generate Grad-CAM visualizations.")
    
    else:
        # Preprocess the image for Grad-CAM processing
        img_resized = cv2.resize(st.session_state.image_array, (224, 224))
        img_array = np.expand_dims(img_resized.astype("float32") / 255, axis=0)

        # Get model predictions
        preds = model.predict(img_array)[0]
        predicted_class = np.argmax(preds)

        # Generate Grad-CAM heatmap
        heatmap = get_gradcam_heatmap(img_array, predicted_class)

        # Overlay heatmap onto the original image
        superimposed_img = overlay_gradcam(st.session_state.image_array, heatmap)

        # Create side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.image(display_img_rgb, caption="📷 **Original Image**", use_container_width=True )

        with col2:
            st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), 
                     caption="🔥 **Grad-CAM Heatmap**", use_container_width=True)

        # Explanation box with styling
        st.markdown(
            """
            <div style="
                background-color: #6d6d6d;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            ">
                <h4 style="color: #2b2313;">🔎 Understanding Grad-CAM</h4>
                <p>Grad-CAM highlights important regions of the image that influenced the model's decision. The more intense the red regions, 
                the more **significant** they were in classification. This helps us **interpret** the AI model’s decision-making. 🌟</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("---")


# Define storage directory
HISTORY_DIR = r"History/classification_history"
CSV_HISTORY_FILE = os.path.join(HISTORY_DIR, "classification_history.csv")

# Ensure directory exists
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# Load existing history if available
if os.path.exists(CSV_HISTORY_FILE):
    history_df = pd.read_csv(CSV_HISTORY_FILE)
else:
    history_df = pd.DataFrame(columns=["Timestamp", "Filename", "Predicted Family", "Confidence"])

# Tab 5: User Gallery & Classification History
with tabs[4]:
    st.header("🔍 Explore previously classified butterflies! 🦋")

    # Save the current classification if a new image is uploaded
    if "image_array" in st.session_state and st.session_state.image_array is not None:
        img_resized = cv2.resize(st.session_state.image_array, (224, 224))
        img_array = np.expand_dims(img_resized.astype("float32") / 255, axis=0)

        # Predict the butterfly family
        preds = model.predict(img_array)[0]
        sorted_indices = np.argsort(preds)[::-1]
        predicted_class = sorted_indices[0]
        predicted_label = class_labels[predicted_class]
        confidence = preds[predicted_class]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{predicted_label}.png"
        file_path = os.path.join(HISTORY_DIR, filename)

        # Save the image
        img_rgb = st.session_state.image_array
        Image.fromarray(img_rgb).save(file_path)

        # Append new entry to history
        new_entry = pd.DataFrame([[timestamp, filename, predicted_label, f"{confidence:.2f}"]], 
                                 columns=["Timestamp", "Filename", "Predicted Family", "Confidence"])
        history_df = pd.concat([history_df, new_entry], ignore_index=True)
        history_df.to_csv(CSV_HISTORY_FILE, index=False)  # Save updated history

    # Display Classification History Table
    if len(history_df) > 0:
        st.write("### 🏆 Classification History")
        st.dataframe(history_df.drop(columns=["Filename"]), use_container_width=True)

        # Filter selection for search
        selected_family = st.selectbox("📌 Filter by Butterfly Family:", ["All"] + history_df["Predicted Family"].unique().tolist())
        if selected_family != "All":
            filtered_df = history_df[history_df["Predicted Family"] == selected_family]
        else:
            filtered_df = history_df

        # Display Image Gallery
        st.write("### 🖼️ Classification Gallery")
        cols = st.columns(3)  # Create 3-column layout for gallery
        for index, row in filtered_df.iterrows():
            with cols[index % 3]:
                img_path = os.path.join(HISTORY_DIR, row["Filename"])
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{row['Predicted Family']} ({row['Confidence']}%)", width=200)

        # Allow users to re-run classification on past images
        selected_index = st.selectbox("📸 Select an image to re-run classification:", filtered_df.index)
        if st.button("🔄 Reclassify Selected Image"):
            selected_entry = filtered_df.iloc[selected_index]
            img_path = os.path.join(HISTORY_DIR, selected_entry["Filename"])
            if os.path.exists(img_path):
                st.session_state.image_array = cv2.imread(img_path)
                st.success(f"✅ Reloaded image from {selected_entry['Timestamp']} for classification!")

        # Allow users to download history as CSV
        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Classification History",
            data=csv_buffer.getvalue(),
            file_name="classification_history.csv",
            mime="text/csv"
        )

    else:
        st.info("📌 **No classification history available.** Start classifying images to track your progress!")

# Tab 6: About & Contact
with tabs[5]:
    st.header("📌 About & Contact")
    
    # Create a two-column layout for better structure
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(r"volumetric-drop-with-ecosystem-inside-generative-ai.jpg", caption="🐛 ➡️ 🦋 AI for Conservation", use_container_width =True)

    with col2:
        st.subheader("🌿 About FloraFly")
        st.write(
        "**FloraFly** is an AI-powered butterfly classification tool designed to assist researchers, conservationists, "
        "and nature enthusiasts in **accurately identifying butterfly families**. "
        "Powered by **state-of-the-art deep learning**, FloraFly bridges the gap between technology and biodiversity, "
        "helping to advance conservation efforts and ecological research. 🦋✨"
        )


    # Divider for clean separation
    st.markdown("---")

    # Contact Section with attractive layout
    st.subheader("📬 Get in Touch!")
    st.markdown("""
        <div style="
            background-color: #6d6d6d;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        ">
            <h4>📧 Email:</h4>
            <p>support@florafly.com</p>
            <h4>🌍 Website:</h4>
            <p><a href='http://florafly.com' target='_blank'>FloraFly.com</a></p>
            <h4>📱 Follow Us:</h4>
            <p>🐦 Twitter: <a href='https://twitter.com/floraflyAI' target='_blank'>@FloraFlyAI</a></p>
            <p>📷 Instagram: <a href='https://instagram.com/floraflyAI' target='_blank'>@FloraFlyAI</a></p>
        </div>
    """, unsafe_allow_html=True)

    # Final Call-to-Action (CTA)
    st.markdown("---")
    st.write("🚀 **Join us in using AI to conserve nature—one butterfly at a time!** 🦋")

