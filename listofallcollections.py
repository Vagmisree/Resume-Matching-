import httpx
import uuid
from sentence_transformers import SentenceTransformer

# --- Configuration ---
QDRANT_URL = "https://qdrant-az-dev.smartx.services"
API_KEY = "smartx-dev"
COLLECTION_NAME = "1my_collection"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

# --- Sentence Embedding Model ---
model = SentenceTransformer("intfloat/e5-small-v2")


# --- 1. List all collections ---
def list_collections():
    url = f"{QDRANT_URL}/collections"
    response = httpx.get(url, headers=HEADERS)
    if response.status_code == 200:
        collections = response.json()["result"]["collections"]
        print("✅ Collections:")
        for col in collections:
            print(f" - {col['name']}")
        return collections
    else:
        print("❌ Failed to list collections:", response.text)
        return []


# --- 2. Create collection if not exists ---
def create_collection_if_needed():
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
    payload = {
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        }
    }
    response = httpx.put(url, headers=HEADERS, json=payload)
    if response.status_code == 200:
        print(f"✅ Collection '{COLLECTION_NAME}' created or already exists.")
    else:
        print("❌ Failed to create collection:", response.text)


# --- 3. Upload new JDs ---
def upload_jds(jd_list):
    vectors = model.encode(jd_list).tolist()
    points = []
    for jd, vec in zip(jd_list, vectors):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {"text": jd}
        })

    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true"
    response = httpx.put(url, headers=HEADERS, json={"points": points})

    if response.status_code == 200:
        print(f"✅ Uploaded {len(jd_list)} JDs to collection '{COLLECTION_NAME}'.")
    else:
        print("❌ Upload failed:", response.status_code, response.text)


# --- 4. Fetch uploaded JDs from collection ---
def fetch_uploaded_jds(limit=10):
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll"
    payload = {
        "limit": limit,
        "with_payload": True,
        "with_vector": False
    }
    response = httpx.post(url, headers=HEADERS, json=payload)
    if response.status_code == 200:
        points = response.json()["result"]["points"]
        print(f"✅ Retrieved {len(points)} JDs from '{COLLECTION_NAME}':")
        for pt in points:
            print(" -", pt["payload"]["text"][:60], "...")
    else:
        print("❌ Failed to fetch points:", response.text)


# --- Run Full Flow ---
if __name__ == "__main__":
    # Sample JD data
    jd_samples = [
        "Looking for a Python Developer with experience in FastAPI and Docker.",
        "We need a React Native developer who knows Expo and NativeWind.",
        "Seeking a data scientist with hands-on ML project experience and Pandas/NumPy skills.",
        "Backend engineer required with strong knowledge in Node.js and MongoDB.",
        "DevOps Engineer familiar with AWS, Kubernetes, and CI/CD tools like GitHub Actions."
    ]

    print("\n--- Step 1: List existing collections ---")
    list_collections()

    print("\n--- Step 2: Create collection if needed ---")
    create_collection_if_needed()

    print("\n--- Step 3: Upload sample JDs ---")
    upload_jds(jd_samples)

    print("\n--- Step 4: Fetch uploaded JDs to confirm ---")
    fetch_uploaded_jds(limit=10)
