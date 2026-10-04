import os

MODEL_NAME = 'resnet50'
INPUT_SHAPE = (224, 224, 3)
POOLING = 'avg'
DATA_DIR = 'data'
UPLOADS_DIR = 'uploads'
SAMPLE_DIR = 'sample'
EMBEDDINGS_FILE = 'embedding.pkl'
FILENAMES_FILE = 'filenames.pkl'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
SIMILARITY_THRESHOLD = 0.4
TOP_N_MATCHES = 3
