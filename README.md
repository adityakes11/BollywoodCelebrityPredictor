# Bollywood Celebrity Predictor 🎬

An AI-powered web application that takes an uploaded image of a person, detects their face, and compares it against a database of Bollywood celebrities to find their closest lookalike.

## 🧠 Architecture & Tech Stack

This project is an end-to-end Machine Learning pipeline demonstrating Computer Vision and Deep Learning techniques:

*   **Python, TensorFlow & Keras:** Core deep learning frameworks used.
*   **MTCNN (Multi-task Cascaded Convolutional Networks):** Used for robust, real-time facial detection and bounding-box extraction from user-uploaded images.
*   **ResNet50 (VGGFace):** Applied **Transfer Learning** by utilizing this pre-trained CNN as a feature extractor to generate 2048-dimensional facial embeddings.
*   **Scikit-Learn (Cosine Similarity):** Optimized the matching algorithm using vectorized cosine distance to instantly retrieve the closest matches from the database.
*   **OpenCV & Pillow:** For image preprocessing, color-space conversion (BGR to RGB), and resizing.
*   **Streamlit:** Designed and deployed an interactive UI, integrating caching mechanisms (`@st.cache_resource`) to prevent model reloading and improve app performance.

## 🚀 Key Highlights (For Resume/Portfolio)
If you are looking at this project from a portfolio perspective, here is what it demonstrates:
*   **Practical Deep Learning:** Shows the ability to leverage heavy, pre-trained CNN architectures for real-world tasks (Feature Extraction) without requiring massive compute for training from scratch.
*   **Computer Vision (CV) Skills:** Proves an understanding of processing images and using specialized CV models for object (face) detection.
*   **Math into Code:** Successfully applied mathematical concepts (Cosine Similarity) to solve a search/matching problem.
*   **Product Engineering:** Built a functional UI around the AI model, proving the ability to build tools that non-technical users can interact with.

---

## 🛠️ Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Data:**
   Put your celebrity images inside the `data` folder structured like this:
   ```text
   data/
       Actor_Name_1/
           img1.jpg
           img2.jpg
       Actor_Name_2/
           img1.jpg
   ```

3. **Extract Features (One-time step):**
   Run the feature extractor to generate embeddings for all images in your dataset:
   ```bash
   python feature_extractor.py
   ```
   *This will generate `embedding.pkl` and `filenames.pkl` in the root directory.*

4. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration
You can edit `config.py` to change settings like the number of top matches (`TOP_N_MATCHES`), similarity threshold, and directory paths.

## ▶️ Run locally with Docker (optional)

```bash
# Build the Docker image
docker build -t bollywood-predictor .

# Run the container and expose Streamlit port
docker run -p 8501:8501 bollywood-predictor
```
