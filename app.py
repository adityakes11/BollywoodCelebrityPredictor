import streamlit as st
import os
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
import config
import utils

st.set_page_config(page_title="Bollywood Celebrity Match", page_icon="🎬", layout="wide")

# -----------------------------
# Caching Model and Data
# -----------------------------
@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(
        model=config.MODEL_NAME,
        include_top=False,
        input_shape=config.INPUT_SHAPE,
        pooling=config.POOLING
    )
    return detector, model

@st.cache_data
def get_data():
    return utils.load_data()

# -----------------------------
# App State & Setup
# -----------------------------
detector, model = load_models()
feature_list, filenames = get_data()

if feature_list is None or filenames is None:
    st.error(f"❌ Could not load data. Ensure {config.EMBEDDINGS_FILE} and {config.FILENAMES_FILE} exist by running feature_extractor.py.")
    st.stop()

def save_uploaded_image(uploaded_image):
    try:
        # Validate extension
        ext = uploaded_image.name.split('.')[-1].lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            st.error(f"❌ Unsupported file format: {ext}. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}")
            return None

        if not os.path.exists(config.UPLOADS_DIR):
            os.makedirs(config.UPLOADS_DIR)

        file_path = os.path.join(config.UPLOADS_DIR, uploaded_image.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving file: {e}")
        return None

# -----------------------------
# Streamlit UI Sidebar
# -----------------------------
st.sidebar.title("Instructions 📝")
st.sidebar.markdown(f"""
1. Upload an image of yourself or choose a sample.
2. We detect your face and extract features.
3. We compare you against a database of Bollywood celebrities.
4. Top {config.TOP_N_MATCHES} closest matches are shown!

**Tips:**
- Ensure good lighting 💡
- Look straight into the camera 📷
- Only 1 face per image for best results.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Try a Sample Image")
sample_files = os.listdir(config.SAMPLE_DIR) if os.path.exists(config.SAMPLE_DIR) else []
selected_sample = st.sidebar.selectbox("Choose a sample", ["None"] + sample_files)

# -----------------------------
# Streamlit UI Main Page
# -----------------------------
st.title("🎬 Which Bollywood Celebrity Are You?")

uploaded_image = st.file_uploader("Upload your image", type=config.ALLOWED_EXTENSIONS)

file_path = None
display_image = None

if uploaded_image is not None:
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        display_image = Image.open(uploaded_image)
elif selected_sample != "None":
    file_path = os.path.join(config.SAMPLE_DIR, selected_sample)
    display_image = Image.open(file_path)

if file_path is not None and display_image is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(display_image, caption="Input Image", width=300)

    with col2:
        with st.spinner("Analyzing face and searching database..."):
            features = utils.extract_features(file_path, model, detector)

        if features is None:
            st.error("❌ No face detected. Please upload a clear image of a face.")
        else:
            indices, scores = utils.recommend(feature_list, features, top_n=config.TOP_N_MATCHES)

            if scores[0] < config.SIMILARITY_THRESHOLD:
                st.warning(f"⚠️ Best match has a low similarity score ({scores[0]:.2f}). Results may not be accurate.")
            else:
                st.success("✅ Matches Found!")

            st.header("Top Matches")
            match_cols = st.columns(len(indices))
            
            for i, (idx, score) in enumerate(zip(indices, scores)):
                actor_name = utils.get_actor_name(filenames[idx])
                
                with match_cols[i]:
                    st.subheader(f"#{i+1}: {actor_name}")
                    st.markdown(f"**Similarity:** {score*100:.1f}%")
                    try:
                        st.image(filenames[idx], width=250)
                    except:
                        st.error("Image file missing")