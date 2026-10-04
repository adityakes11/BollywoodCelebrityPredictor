from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN
import os
import config
import utils

# Load Data
feature_list, filenames = utils.load_data()
if feature_list is None:
    print(f"Error: Could not load {config.EMBEDDINGS_FILE}. Please run feature_extractor.py first.")
    exit(1)

# Load Models
model = VGGFace(
    model=config.MODEL_NAME,
    include_top=False,
    input_shape=config.INPUT_SHAPE,
    pooling=config.POOLING
)
detector = MTCNN()

# Process Sample Image
sample_image_path = os.path.join(config.SAMPLE_DIR, 'satya.jpg')
print(f"Processing image: {sample_image_path}")

features = utils.extract_features(sample_image_path, model, detector)

if features is None:
    print("Error: No face detected in the sample image.")
else:
    # Find Best Matches
    indices, scores = utils.recommend(feature_list, features, top_n=1)
    best_match_idx = indices[0]
    best_match_score = scores[0]
    
    predicted_actor = utils.get_actor_name(filenames[best_match_idx])
    print(f"Best match: {predicted_actor} (Similarity: {best_match_score:.4f})")

    # Display the result
    temp_img = cv2.imread(filenames[best_match_idx])
    if temp_img is not None:
        cv2.imshow(f'Match: {predicted_actor}', temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load the image {filenames[best_match_idx]}")