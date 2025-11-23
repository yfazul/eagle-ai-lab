# image_ingest.py  (FINAL 1536-dim VERSION)

import os
import base64
from dotenv import load_dotenv
import chromadb

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# -------------------------------------
# ENV
# -------------------------------------
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY   = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT    = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE  = os.getenv("CHROMA_DATABASE")

PAGE64_DIR = "./data/page64_images"

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------
# Chroma (Cloud)
# -------------------------------------
chroma = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# Delete old collection
try:
    chroma.delete_collection("tn_page64_images")
except:
    pass

# Create new collection (1536-dim expected)
collection = chroma.create_collection(
    name="tn_page64_images",
    metadata={"hnsw:space": "cosine"}
)

# -------------------------------------
# Embeddings (1536-dimensional)
# -------------------------------------
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dims
    openai_api_key=OPENAI_API_KEY,
)

# -------------------------------------
# Caption Generator
# -------------------------------------
def generate_caption(image_path: str) -> str:
    """
    Produce a short caption for the image using GPT-4o-mini vision.
    Fully compatible with the latest OpenAI Python SDK (2025 update).
    """
    import base64

    # Read + encode image
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one short, simple sentence suitable for school children."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"}
                        }
                    ]
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        # -----------------------------------------
        # CASE 1: content is a simple string
        # -----------------------------------------
        if isinstance(content, str):
            return content.strip()

        # -----------------------------------------
        # CASE 2: content is a list of content blocks
        # -----------------------------------------
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "type") and block.type == "text":
                    text_parts.append(block.text)

            if text_parts:
                return text_parts[0].strip()

        # fallback
        return "Image from page 64"

    except Exception as e:
        print(f"❌ Caption error for {image_path}: {e}")
        return "Image from page 64"


# -------------------------------------
# MAIN INGESTION
# -------------------------------------
print("📥 Ingesting Page-64 images...\n")

image_files = [
    f for f in os.listdir(PAGE64_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not image_files:
    print("❌ No images found in ./data/page64_images")
    exit()

counter = 1

for fname in sorted(image_files):
    img_path = os.path.join(PAGE64_DIR, fname)

    print(f"🖼 Processing: {img_path}")

    # Generate caption
    caption = generate_caption(img_path)

    # Compute embedding
    embedding = embedder.embed_query(caption)

    # Document ID
    doc_id = f"img64_{counter}"

    # Store in Chroma
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[caption],
        metadatas=[{
            "page": 64,
            "filename": fname,
            "type": "image"
        }]
    )

    print(f"   → Added as {doc_id}")
    counter += 1

print("\n=======================================")
print(f"🎉 Ingestion Complete! {counter-1} images added.")
print("=======================================\n")
