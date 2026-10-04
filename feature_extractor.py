import os
import numpy as np
import pickle
from tqdm import tqdm
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
import config

def extract_features_bulk(img_path, model):
    try:
        img = image.load_img(img_path, target_size=config.INPUT_SHAPE[:2])
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img, verbose=0).flatten()
        return result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(config.DATA_DIR):
        print(f"Error: Data directory '{config.DATA_DIR}' not found.")
        exit(1)

    print("Gathering image filenames...")
    actors = os.listdir(config.DATA_DIR)
    filenames = []

    for actor in actors:
        actor_dir = os.path.join(config.DATA_DIR, actor)
        if not os.path.isdir(actor_dir):
            continue
        for file in os.listdir(actor_dir):
            if file.lower().endswith(tuple(config.ALLOWED_EXTENSIONS)):
                filenames.append(os.path.join(actor_dir, file))

    print(f"Found {len(filenames)} images.")

    if len(filenames) == 0:
        print("No images found to process.")
        exit(1)

    print("Loading VGGFace model...")
    model = VGGFace(
        model=config.MODEL_NAME,
        include_top=False,
        input_shape=config.INPUT_SHAPE,
        pooling=config.POOLING
    )

    print("Extracting features...")
    features = []
    valid_filenames = []

    for file in tqdm(filenames):
        feature = extract_features_bulk(file, model)
        if feature is not None:
            features.append(feature)
            valid_filenames.append(file)

    print(f"Successfully extracted features for {len(features)} images.")

    print("Saving filenames and embeddings...")
    pickle.dump(valid_filenames, open(config.FILENAMES_FILE, 'wb'))
    pickle.dump(features, open(config.EMBEDDINGS_FILE, 'wb'))
    
    print("Done!")
