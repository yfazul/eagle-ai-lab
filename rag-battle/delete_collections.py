# delete_old_collections.py

import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

collections = ["tn_class6_text", "tn_class6_images"]

for name in collections:
    try:
        client.delete_collection(name)
        print(f"Deleted collection: {name}")
    except Exception as e:
        print(f"Could not delete {name}: {e}")

print("Step 1 completed: Old collections removed.")
