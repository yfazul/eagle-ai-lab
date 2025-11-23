# query_page_55.py

import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

collection = client.get_collection("tn_class6_text")

results = collection.get(
    where={"page": 55},
    include=["documents", "metadatas"]  # ⬅ FIXED: no 'ids'
)

print("======================================")
print("📄 Chroma Results for Page 55")
print("======================================")

ids = results["ids"]
docs = results["documents"]
metas = results["metadatas"]

if ids:
    for i in range(len(ids)):
        print(f"\n--- Chunk {i+1} | ID: {ids[i]} ---")
        print(f"Page: {metas[i]['page']}")
        print(docs[i])
else:
    print("⚠️ No chunks found for page 55.")
