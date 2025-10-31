import os
import torch
import timm
import faiss
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms
import easyocr
import pickle
import base64
import requests
from serpapi.google_search import GoogleSearch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO
import uuid  # For unique temp filenames

# --- Flask Imports ---
from flask import Flask, request, jsonify, render_template, send_from_directory

# ------------------- CONFIG -------------------
# !!! IMPORTANT: Update this path to your database !!!
DATABASE_FOLDER = "/media/manthan/805B-45C6/PycharmProjects/JupyterProject/CYLOTROPIA/hackx/front/database"
SERPAPI_KEY = "bf5636512c97bb0aa0843ed60a84c357d611c49cfd61c8c1bc54a775e19e13e0"
# Create a folder to store temporary uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- DEEPFAKE DETECTION -------------------
print("Loading deepfake detection model...")
df_processor = AutoImageProcessor.from_pretrained("umm-maybe/ai-image-detector")
df_model = AutoModelForImageClassification.from_pretrained("umm-maybe/ai-image-detector")


def detect_deepfake(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = df_processor(images=image, return_tensors="pt")
        outputs = df_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class_id = torch.max(probs, 1)
        label = df_model.config.id2label[predicted_class_id.item()]
        return label, float(confidence)
    except Exception as e:
        print(f"Deepfake detection error: {e}")
        return "Unknown", 0.0


# ------------------- Feature Extractor -------------------
class FeatureExtractor:
    def __init__(self, model_name='resnet18', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.model.to(self.device)
        config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**config, is_training=False)
        self.ocr_reader = easyocr.Reader(['en'])
        print("Feature extractor and OCR reader loaded.")

    def extract(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(img_tensor)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def extract_text(self, image_path: str):
        try:
            results = self.ocr_reader.readtext(image_path)
            detected_text = " ".join([text for _, text, _ in results])
            return detected_text if detected_text.strip() else "No text found"
        except Exception as e:
            print(f"OCR error on image {image_path}: {e}")
            return "No text found"


# ------------------- Error Level Analysis (ELA) -------------------
def ela_detect(image_path, scale=15):
    try:
        img = Image.open(image_path).convert("RGB")
        with BytesIO() as tmp_buffer:
            img.save(tmp_buffer, "JPEG", quality=90)
            tmp_buffer.seek(0)
            ela_img = Image.open(tmp_buffer)
            diff = ImageChops.difference(img, ela_img)

        diff = ImageEnhance.Brightness(diff).enhance(scale)
        ela_np = np.array(diff)
        score = np.mean(ela_np) / 255.0
        return diff, round(score, 3)  # Return PIL image object
    except Exception as e:
        print(f"ELA error on {image_path}: {e}")
        return None, 0.0


# ------------------- FAISS Index -------------------
def build_faiss_index(feature_extractor, image_folder: str, index_file="image_features.index",
                      paths_file="image_paths.pkl"):
    if not os.path.isdir(image_folder):
        print(f"Error: Database folder not found at {image_folder}")
        return None, None

    if os.path.exists(index_file) and os.path.exists(paths_file):
        print("Loading saved FAISS index and image paths...")
        index = faiss.read_index(index_file)
        with open(paths_file, "rb") as f:
            image_paths = pickle.load(f)
        return index, image_paths

    print("Building FAISS index from scratch...")
    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(image_paths)} images to index.")

    all_features, valid_paths = [], []
    for path in image_paths:
        features = feature_extractor.extract(path)
        if features is not None:
            all_features.append(features)
            valid_paths.append(path)

    if not all_features:
        print("No features extracted. Exiting.")
        return None, None

    features_np = np.array(all_features).astype('float32')
    d = features_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features_np)
    print(f"FAISS index built with {index.ntotal} vectors.")

    faiss.write_index(index, index_file)
    with open(paths_file, "wb") as f:
        pickle.dump(valid_paths, f)

    return index, valid_paths


# ------------------- Local Similarity Search -------------------
def search_similar_images(feature_extractor, index, image_paths, query_image_path, k=5):
    query_features = feature_extractor.extract(query_image_path)
    if query_features is None:
        return [], [], "No text found", 0.0, None

    query_features_np = np.array([query_features]).astype('float32')
    distances, indices = index.search(query_features_np, k)
    similar_paths = [image_paths[i] for i in indices[0]]
    query_text = feature_extractor.extract_text(query_image_path)
    ela_image, tamper_score = ela_detect(query_image_path)

    return similar_paths, distances[0], query_text, tamper_score, ela_image


# ------------------- SerpAPI Online Search -------------------
def serpapi_search(query_text):
    try:
        params = {"engine": "bing", "q": query_text, "api_key": SERPAPI_KEY, "num": "10"}
        search = GoogleSearch(params)
        results = search.get_dict()
        web_results = results.get("organic_results", [])
        return [{"title": r.get("title", ""), "link": r.get("link", ""), "snippet": r.get("snippet", "")} for r in
                web_results]
    except Exception as e:
        print(f"Error during SerpAPI search: {e}")
        return []


def serpapi_image_lookup(image_path):
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        params = {"engine": "bing_images", "api_key": SERPAPI_KEY, "image_content": img_b64}
        search = GoogleSearch(params)
        results = search.get_dict()
        image_results = results.get("images_results", [])
        return [r.get("source") or r.get("link") for r in image_results[:5] if r.get("source") or r.get("link")]
    except Exception as e:
        print(f"Reverse image search failed: {e}")
        return []


# ------------------- Image Encoding Helper -------------------
def encode_image_to_base64(image_path=None, pil_image=None):
    """Encodes an image (from path or PIL object) to a base64 string."""
    try:
        if image_path:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif pil_image:
            buffer = BytesIO()
            pil_image.convert("RGB").save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
    return None


# ------------------- Fact Checking Wrapper (API Version) -------------------
def analyze_image_pipeline(query_image_path, k=4):
    """
    Runs the full analysis and returns a dictionary of results.
    This function replaces fact_check_image and display_results.
    """
    print(f"\n🧠 Analyzing {query_image_path}...")
    results = {}

    # 1. Local Similarity, OCR, and ELA
    similar_paths, distances, query_text, tamper_score, ela_image_pil = search_similar_images(
        extractor, faiss_index, db_image_paths, query_image_path, k
    )
    results['query_text'] = query_text
    results['tamper_score'] = tamper_score
    results['ela_image_b64'] = encode_image_to_base64(pil_image=ela_image_pil)

    # Encode similar images
    results['similar_images_b64'] = [encode_image_to_base64(path) for path in similar_paths]

    # 2. Deepfake Detection
    print("🔬 Running deepfake detection...")
    deepfake_label, confidence = detect_deepfake(query_image_path)
    results['deepfake_label'] = deepfake_label
    results['deepfake_confidence'] = confidence * 100  # As percentage

    # 3. Online Search
    web_results = []
    visual_links = []
    if query_text == "No text found":
        print("⚠️ No text detected — running visual reverse image search...")
        visual_links = serpapi_image_lookup(query_image_path)
    else:
        print("🌐 Searching for online context...")
        web_results = serpapi_search(query_text)
        if not web_results:
            print("No text-based results. Trying visual lookup...")
            visual_links = serpapi_image_lookup(query_image_path)

    results['web_results'] = web_results
    results['visual_links'] = visual_links

    # 4. Final Verdict
    summary_verdict = "🟢 Image appears authentic."
    if tamper_score > 0.3:
        summary_verdict = "⚠️ Possible manipulation detected (High ELA score)."
    elif "fake" in deepfake_label.lower() and confidence > 0.7:
        summary_verdict = "⚠️ Likely AI-generated content (Model is confident)."

    results['summary_verdict'] = summary_verdict
    print("✅ Analysis complete.")
    return results


# ------------------- FLASK APP -------------------
app = Flask(__name__, template_folder='.')


@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyzes the uploaded image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    # Save the file temporarily
    temp_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        file.save(temp_path)

        # Run the full analysis pipeline
        analysis_results = analyze_image_pipeline(temp_path)

        # Add the query image to the results
        analysis_results['query_image_b64'] = encode_image_to_base64(image_path=temp_path)

        return jsonify(analysis_results)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ------------------- MAIN -------------------
if __name__ == "__main__":
    # --- CORRECTED CHECK ---
    # This check now just ensures the key is not empty.
    if not SERPAPI_KEY:
        print("Error: SERPAPI_KEY not set in app.py")
    # -------------------------
    elif not os.path.isdir(DATABASE_FOLDER):
        print(f"Error: DATABASE_FOLDER not found at {DATABASE_FOLDER}")
    else:
        # Load models globally
        print("Initializing global models...")
        extractor = FeatureExtractor()
        faiss_index, db_image_paths = build_faiss_index(extractor, DATABASE_FOLDER)

        if faiss_index is None or db_image_paths is None:
            print("Failed to load or build FAISS index. Exiting.")
        else:
            print("\n✅ Server ready. Starting Flask app...")
            # Host='0.0.0.0' makes it accessible on your network
            app.run(debug=True, host='0.0.0.0', port=5000)