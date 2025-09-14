import httpx
import json

QDRANT_URL = "https://qdrant-az-dev.smartx.services"
API_KEY = "smartx-dev"
COLLECTION_NAME = "OpenembdJDs"

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

collection_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"

payload = {
    "vectors": {
        "size": 384,
        "distance": "Cosine"
    }
}

try:
    # Use PUT to create or recreate the collection
    response = httpx.put(collection_url, headers=headers, json=payload, timeout=10)
    
    if response.status_code == 200:
        print(f"✅ Collection '{COLLECTION_NAME}' created or updated successfully.")
    else:
        print(f"❌ Failed to create collection '{COLLECTION_NAME}': {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ HTTP request failed: {e}")
