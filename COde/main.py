import os
import torch
import timm
import faiss
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms
import matplotlib.pyplot as plt
import easyocr
import pickle
import base64
import requests
from serpapi.google_search import GoogleSearch

from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO

SERPAPI_KEY = "bf5636512c97bb0aa0843ed60a84c357d611c49cfd61c8c1bc54a775e19e13e0"

# ------------------- DEEPFAKE DETECTION -------------------
print("Loading deepfake detection model...")
# <-- CHANGED: Using a model trained for deepfake detection
processor = AutoImageProcessor.from_pretrained("umm-maybe/ai-image-detector")
model = AutoModelForImageClassification.from_pretrained("umm-maybe/ai-image-detector")


def detect_deepfake(image_path):
    """Detects if an image is AI-generated or real using a transformer model."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # <-- CHANGED: This model's output is [fake, real]
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the highest probability and its class ID
        confidence, predicted_class_id = torch.max(probs, 1)

        # Get the label string (e.g., "fake" or "real")
        label = model.config.id2label[predicted_class_id.item()]

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
        # Use BytesIO for in-memory temporary file
        with BytesIO() as tmp_buffer:
            img.save(tmp_buffer, "JPEG", quality=90)
            tmp_buffer.seek(0)
            ela_img = Image.open(tmp_buffer)
            diff = ImageChops.difference(img, ela_img)

        diff = ImageEnhance.Brightness(diff).enhance(scale)
        ela_np = np.array(diff)
        score = np.mean(ela_np) / 255.0
        return diff, round(score, 3)
    except Exception as e:
        print(f"ELA error on {image_path}: {e}")
        return None, 0.0


# ------------------- FAISS Index -------------------
def build_faiss_index(feature_extractor, image_folder: str, index_file="image_features.index",
                      paths_file="image_paths.pkl"):
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
        print("Could not extract features from the query image.")
        return [], [], "No text found", 0.0, None

    query_features_np = np.array([query_features]).astype('float32')
    distances, indices = index.search(query_features_np, k)
    similar_paths = [image_paths[i] for i in indices[0]]

    query_text = feature_extractor.extract_text(query_image_path)
    ela_image, tamper_score = ela_detect(query_image_path)

    # <-- CHANGED: Return the ELA image object, don't save it
    return similar_paths, distances[0], query_text, tamper_score, ela_image


# ------------------- SerpAPI Online Search -------------------
def serpapi_search(query_text):
    try:
        params = {
            "engine": "bing",
            "q": query_text,
            "api_key": SERPAPI_KEY,
            "num": "10"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        web_results = results.get("organic_results", [])
        return [
            {"title": r.get("title", ""), "link": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in web_results
        ]
    except Exception as e:
        print(f"Error during SerpAPI search: {e}")
        return []


def serpapi_image_lookup(image_path):
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        params = {
            "engine": "bing_images",
            "api_key": SERPAPI_KEY,
            "image_content": img_b64
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        image_results = results.get("images_results", [])
        return [r.get("source") or r.get("link") for r in image_results[:5] if r.get("source") or r.get("link")]
    except Exception as e:
        print(f"Reverse image search failed: {e}")
        return []


# ------------------- Display -------------------
# <-- CHANGED: Reworked for better layout and to accept ela_image object
def display_results(query_path, result_paths, query_text=None, tamper_score=None, ela_image=None):
    num_results = len(result_paths)
    # Set columns to be the max of 2 (for Query/ELA) or num_results
    cols = max(num_results, 2)
    plt.figure(figsize=(15, 8))  # Adjusted size

    # --- Top Row: Query Image ---
    plt.subplot(2, cols, 1)
    plt.imshow(Image.open(query_path))
    title = "Query Image"
    if query_text:
        title += f"\nText: {query_text}"
    if tamper_score is not None:
        title += f"\nTamper Score: {tamper_score}"
    plt.title(title)
    plt.axis('off')

    # --- Top Row: ELA Analysis ---
    if ela_image is not None:
        plt.subplot(2, cols, 2)
        plt.imshow(ela_image)
        plt.title("ELA Analysis")
        plt.axis('off')

    # --- Bottom Row: Similar Images ---
    for i, path in enumerate(result_paths):
        # Plot results starting at the first position of the second row
        plt.subplot(2, cols, cols + i + 1)
        plt.imshow(Image.open(path))
        plt.title(f"Result {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ------------------- Fact Checking Wrapper -------------------
def fact_check_image(feature_extractor, index, image_paths, query_image_path, k=5):
    print("\n🧠 Running hybrid image fact-checking pipeline...")

    # <-- CHANGED: Capture the new ela_image object
    similar_paths, distances, query_text, tamper_score, ela_image = search_similar_images(
        feature_extractor, index, image_paths, query_image_path, k
    )

    print(f"\nDetected text: {query_text}")
    print(f"ELA Manipulation Likelihood: {tamper_score}")

    # 🔍 Deepfake Detection
    print("\n🔬 Running deepfake detection...")
    deepfake_label, confidence = detect_deepfake(query_image_path)
    print(f"Deepfake Model Verdict: {deepfake_label.upper()} ({confidence * 100:.2f}% confidence)")

    if query_text == "No text found":
        print("\n⚠️ No text detected — running visual reverse image search...")
        links = serpapi_image_lookup(query_image_path)
        for i, link in enumerate(links, 1):
            print(f"Visual Match {i}: {link}")
    else:
        print("\n🌐 Searching for online context...")
        results = serpapi_search(query_text)
        if results:
            for i, r in enumerate(results[:4], 1):
                print(f"\nResult {i}: {r['title']}\n{r['snippet']}\n{r['link']}")
        else:
            print("No text-based results found. Trying visual lookup...")
            links = serpapi_image_lookup(query_image_path)
            for i, link in enumerate(links, 1):
                print(f"Visual Match {i}: {link}")

    print("\n✅ Fact-Check Summary:")
    if tamper_score > 0.3:
        print("⚠️ Possible manipulation detected (High ELA score).")
    # <-- CHANGED: Logic fixed to 'and'
    elif "fake" in deepfake_label.lower() and confidence > 0.7:
        print("⚠️ Likely AI-generated or deepfake content (Model is confident).")
    else:
        print("🟢 Image appears authentic.")

    # <-- CHANGED: Pass ela_image object to display
    display_results(
        query_image_path,
        similar_paths,
        query_text=query_text,
        tamper_score=tamper_score,
        ela_image=ela_image
    )


# ------------------- MAIN -------------------
if __name__ == "__main__":
    # <-- CHANGED: Check for API key at the start
    if not SERPAPI_KEY:
        print("Error: SERPAPI_KEY not found.")
        print("Please create a '.env' file in the same directory and add your key:")
        print("SERPAPI_KEY=your_api_key_here")
    else:
        # --- These paths are examples. You MUST change them. ---
        DATABASE_FOLDER = "/media/manthan/805B-45C6/PycharmProjects/JupyterProject/CYLOTROPIA/hackx/front/database"
        QUERY_IMAGE_PATH = "/media/manthan/805B-45C6/PycharmProjects/JupyterProject/CYLOTROPIA/hackx/360_F_248055188_f58hVlOl3p14S6Y3pHfhgpV2dZmunrtJ.jpg"
        NUM_RESULTS = 4

        # --- Check if paths exist ---
        if not os.path.isdir(DATABASE_FOLDER):
            print(f"Error: Database folder not found at {DATABASE_FOLDER}")
        elif not os.path.exists(QUERY_IMAGE_PATH):
            print(f"Error: Query image not found at {QUERY_IMAGE_PATH}")
        else:
            print("Initializing feature extractor...")
            extractor = FeatureExtractor()

            print("Building/loading FAISS index...")
            faiss_index, db_image_paths = build_faiss_index(extractor, DATABASE_FOLDER)

            if faiss_index is not None and db_image_paths:
                fact_check_image(extractor, faiss_index, db_image_paths, QUERY_IMAGE_PATH, k=NUM_RESULTS)
            else:
                print("Failed to load or build FAISS index.")