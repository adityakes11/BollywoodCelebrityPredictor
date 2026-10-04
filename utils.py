import cv2
import numpy as np
# Preprocess the VGGFace input (mean subtraction, scaling, etc.)
# Using the utility from keras_vggface ensures the same preprocessing as the original model.
from keras_vggface.utils import preprocess_input
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import config

def load_data():
    """Load embedding and filenames data.
    Falls back to alternate filenames (feature.pkl) if the default ones are missing.
    """
    try:
        embeddings_path = config.EMBEDDINGS_FILE
        filenames_path = config.FILENAMES_FILE
        # Fallback names
        if not os.path.exists(embeddings_path) and os.path.exists('feature.pkl'):
            embeddings_path = 'feature.pkl'
        if not os.path.exists(filenames_path) and os.path.exists('feature.pkl'):
            filenames_path = 'feature.pkl'
        feature_list = np.array(pickle.load(open(embeddings_path, 'rb')))
        filenames = pickle.load(open(filenames_path, 'rb'))
        return feature_list, filenames
    except Exception as e:
        print(f"Error loading data (embeddings: {embeddings_path}, filenames: {filenames_path}): {e}")
        return None, None

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize large images to avoid MTCNN slowing down
    max_dim = 1000
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if len(results) == 0:
        return None

    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)
    
    face = img_rgb[y:y+height, x:x+width]
    if face.size == 0:
        return None

    face = cv2.resize(face, config.INPUT_SHAPE[:2])
    face_array = np.asarray(face).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features, top_n=1):
    similarity = cosine_similarity(features.reshape(1, -1), feature_list)[0]
    index_pos = np.argsort(similarity)[::-1][:top_n]
    scores = similarity[index_pos]
    return index_pos, scores

def get_actor_name(filepath):
    # Cross-platform way to get the folder name (actor name)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    return " ".join(parent_dir.split('_'))
