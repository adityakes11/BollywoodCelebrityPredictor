from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# -----------------------------
# Load Face Detector and Model
# -----------------------------
detector = MTCNN()

model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg'
)

# -----------------------------
# Load Embeddings
# -----------------------------
feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# -----------------------------
# Save Uploaded Image
# -----------------------------
def save_uploaded_image(uploaded_image):

    try:

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        file_path = os.path.join("uploads", uploaded_image.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        return file_path

    except:
        return None


# -----------------------------
# Extract Face Features
# -----------------------------
def extract_features(img_path, model, detector):

    img = cv2.imread(img_path)

    if img is None:
        return None

    # Convert BGR -> RGB (IMPORTANT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x, y, width, height = results[0]['box']

    # Fix negative coordinates
    x = max(0, x)
    y = max(0, y)

    face = img[y:y+height, x:x+width]

    if face.size == 0:
        return None

    # Resize
    face = cv2.resize(face, (224,224))

    face_array = np.asarray(face).astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result


# -----------------------------
# Recommend Celebrity
# -----------------------------
def recommend(feature_list, features):

    similarity = []

    for i in range(len(feature_list)):

        similarity.append(
            cosine_similarity(
                features.reshape(1,-1),
                feature_list[i].reshape(1,-1)
            )[0][0]
        )

    index_pos = sorted(
        list(enumerate(similarity)),
        reverse=True,
        key=lambda x:x[1]
    )[0][0]

    return index_pos


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Which Bollywood Celebrity Are You?")

uploaded_image = st.file_uploader("Upload your image")

if uploaded_image is not None:

    file_path = save_uploaded_image(uploaded_image)

    if file_path is not None:

        display_image = Image.open(uploaded_image)

        st.image(display_image, caption="Uploaded Image", width=300)

        with st.spinner("Detecting face and finding celebrity match..."):

            features = extract_features(file_path, model, detector)

        if features is None:

            st.error("❌ No face detected. Please upload a clear face image.")

        else:

            index_pos = recommend(feature_list, features)

            predicted_actor = " ".join(
                filenames[index_pos].split('\\')[1].split('_')
            )

            st.success("✅ Match Found!")

            col1, col2 = st.columns(2)

            with col1:
                st.header("Your Image")
                st.image(display_image, width=300)

            with col2:
                st.header("You look like")
                st.subheader(predicted_actor)
                st.image(filenames[index_pos], width=300)