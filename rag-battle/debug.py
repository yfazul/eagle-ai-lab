import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

image_coll = client.get_collection("tn_class6_images")

results = image_coll.get(
    where={"page": 52},
    include=["metadatas", "documents"]
)

print("Images on page 52:", len(results["ids"]))
for i in range(len(results["ids"])):
    print("----")
    print("ID:", results["ids"][i])
    print("OCR:", results["documents"][i][:200])
    print("Meta:", results["metadatas"][i])
