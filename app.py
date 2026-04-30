import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove # Bringing back our surgical tool!

st.set_page_config(page_title="AI Wardrobe Enterprise", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 3.5rem; color: #00D2FF; text-align: center; font-weight: bold;}
    .sub-title { text-align: center; color: #888; font-style: italic; margin-bottom: 30px;}
    .prediction-box { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #00D2FF;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌐 AI Wardrobe <i>Enterprise Hybrid</i></p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by MobileNetV2 + Subject Isolation</p>', unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    file = st.file_uploader("Drop any high-resolution image here...", type=["jpg", "png", "jpeg"])

if file is not None:
    st.divider()
    img_col, res_col = st.columns(2)
    
    with img_col:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Original Input', use_container_width=True)

    with res_col:
        st.subheader("Deep Convolutional Analysis 🧠")
        
        with st.spinner("Isolating subject and querying ImageNet..."):
            
            # --- THE ULTIMATE HYBRID PREPROCESSING ---
            
            # 1. Strip the tricky background
            img_no_bg = remove(image)
            
            # 2. Create a clean WHITE canvas (Industry standard for e-commerce)
            # ImageNet models love white backgrounds!
            white_canvas = Image.new("RGB", img_no_bg.size, (255, 255, 255))
            white_canvas.paste(img_no_bg, mask=img_no_bg.split()[3])
            
            # Show the cleaned image to the user
            st.image(white_canvas, caption="What the AI sees (Cleaned)", width=200)
            
            # 3. Resize for MobileNetV2 (224x224)
            img_resized = white_canvas.resize((224, 224))
            
            # 4. Convert to Array and Preprocess
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # --- PREDICTION ---
           # --- PREDICTION ---
            predictions = model.predict(img_array)
            
            # CHANGE: Grab the Top 3 guesses instead of just Top 1
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
            
            # Display the #1 Best Guess prominently
            best_guess = decoded_predictions[0]
            class_name = best_guess[1].replace('_', ' ').title() 
            confidence = float(best_guess[2])
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin:0;">Top Match: <span style="color:#00D2FF;">{class_name}</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") 
            st.write(f"**Confidence Level: {confidence * 100:.2f}%**")
            st.progress(confidence)
            
            # NEW: Display the alternatives!
            st.markdown("### Alternative Possibilities:")
            for i in range(1, 3): # Loop through guess #2 and #3
                alt_class = decoded_predictions[i][1].replace('_', ' ').title()
                alt_conf = float(decoded_predictions[i][2]) * 100
                st.write(f"- **{alt_class}** ({alt_conf:.2f}% match)")
            
            if confidence > 0.6:
                st.success("High confidence! Subject isolation was successful.")
            else:
                st.warning("Moderate confidence. The AI is weighing multiple possibilities.")