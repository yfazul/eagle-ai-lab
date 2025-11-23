# verify_ingestion.py

import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph

load_dotenv()

# -----------------------------------------
# Load ENV
# -----------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# -----------------------------------------
# Init clients
# -----------------------------------------
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

neo4j = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# -----------------------------------------
# Verify Chroma
# -----------------------------------------
print("\n=== VERIFYING CHROMA COLLECTIONS ===")

try:
    coll = client.get_collection("tn_class6_text")
    print("✓ Chroma collection exists: tn_class6_text")

    # Check count
    count = coll.count()
    print(f"✓ Number of text chunks in Chroma: {count}")

except Exception as e:
    print(f"❌ Chroma error: {e}")

# Verify embedding dimensions
print("\nChecking embedding dimension...")

sample_embedding = embeddings.embed_query("test embedding dimension")
print(f"Embedding dimension: {len(sample_embedding)}")

if len(sample_embedding) == 1536:
    print("✓ Correct: Embeddings are 1536-dimensional")
else:
    print("❌ ERROR: Embeddings are NOT 1536-dimensional")

# -----------------------------------------
# Test vector search
# -----------------------------------------
print("\n=== TESTING VECTOR RETRIEVAL ===")

vectorstore = Chroma(
    client=client,
    collection_name="tn_class6_text",
    embedding_function=embeddings
)

results = vectorstore.similarity_search("beach cleanliness", k=3)

if results:
    print("✓ Vector search returned results:")
    for r in results:
        page = r.metadata.get("page")
        print(f"  • Page {page} → {r.page_content[:80]}...")
else:
    print("❌ No results from vector search!")

# -----------------------------------------
# Verify Neo4j
# -----------------------------------------
print("\n=== VERIFYING NEO4J ===")

try:
    count = neo4j.query("MATCH (t:TextChunk) RETURN count(t) AS c")[0]["c"]
    print(f"✓ TextChunk nodes in Neo4j: {count}")
except Exception as e:
    print(f"❌ Neo4j error: {e}")

# Check pages 1–60 exist
print("\nChecking pages 1–60 in Neo4j...")

missing_pages = []

for p in range(0, 60):
    q = f"MATCH (t:TextChunk {{page: {p}}}) RETURN count(t) AS c"
    result = neo4j.query(q)[0]["c"]
    if result == 0:
        missing_pages.append(p)

if missing_pages:
    print(f"❌ Missing pages in Neo4j: {missing_pages}")
else:
    print("✓ All pages 1–60 exist in Neo4j")

# -----------------------------------------
# Cross-check Chroma vs Neo4j IDs
# -----------------------------------------
print("\n=== CROSS-CHECKING VECTORSTORE VS NEO4J ===")

neo_ids = {
    record["id"]
    for record in neo4j.query("MATCH (t:TextChunk) RETURN t.id AS id")
}

print(f"Neo4j IDs loaded: {len(neo_ids)}")

# get Chroma ids
try:
    res = coll.get(include=["metadatas"])
    chroma_ids = set(res["ids"])
    print(f"Chroma IDs loaded: {len(chroma_ids)}")
except:
    chroma_ids = set()

missing_in_chroma = neo_ids - chroma_ids
missing_in_neo = chroma_ids - neo_ids

if not missing_in_chroma:
    print("✓ All Neo4j nodes exist in Chroma")
else:
    print(f"❌ Missing in Chroma: {list(missing_in_chroma)[:10]}")

if not missing_in_neo:
    print("✓ All Chroma entries exist in Neo4j")
else:
    print(f"❌ Missing in Neo4j: {list(missing_in_neo)[:10]}")

print("\n=== VERIFICATION COMPLETE ===")
