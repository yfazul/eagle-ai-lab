"""
Chroma Data Diagnostic Tool
Checks your Chroma collections for data quality issues
"""

import os
from dotenv import load_dotenv
import chromadb
from collections import Counter

load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

print("="*60)
print("CHROMA DATA DIAGNOSTIC TOOL")
print("="*60)

# Connect to Chroma
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# Get text collection
col = client.get_collection("tn_class6_text")

print("\n=== COLLECTION INFO ===")
print(f"Collection: {col.name}")
print(f"Total documents: {col.count()}")

# Sample some documents to check metadata
print("\n=== SAMPLING METADATA ===")
sample = col.get(limit=50, include=["metadatas"])

if sample and sample.get("metadatas"):
    metas = sample["metadatas"]
    
    # Check page number types
    page_types = Counter()
    page_values = []
    
    for meta in metas:
        if meta and "page" in meta:
            page = meta["page"]
            page_types[type(page).__name__] += 1
            page_values.append(page)
    
    print(f"\nPage metadata types found:")
    for ptype, count in page_types.items():
        print(f"  {ptype}: {count} documents")
    
    print(f"\nSample page values (first 10):")
    for i, page in enumerate(page_values[:10]):
        print(f"  {i+1}. {repr(page)} (type: {type(page).__name__})")
    
    # Check for specific page
    print("\n=== CHECKING PAGE 54 ===")
    
    # Try as string
    results_str = col.get(where={"page": "54"}, include=["documents", "metadatas"])
    print(f"Results with page='54' (string): {len(results_str.get('documents', []))} documents")
    
    # Try as integer
    results_int = col.get(where={"page": 54}, include=["documents", "metadatas"])
    print(f"Results with page=54 (integer): {len(results_int.get('documents', []))} documents")
    
    # Show first result if found
    if results_str.get("documents"):
        print(f"\nFirst document from page 54:")
        print(f"Preview: {results_str['documents'][0][:200]}...")
        print(f"Metadata: {results_str['metadatas'][0]}")
    elif results_int.get("documents"):
        print(f"\nFirst document from page 54:")
        print(f"Preview: {results_int['documents'][0][:200]}...")
        print(f"Metadata: {results_int['metadatas'][0]}")
    else:
        print("\n⚠️ NO DOCUMENTS FOUND FOR PAGE 54!")
        print("Checking what pages DO exist...")
        
        # Get all unique page values
        all_metas = col.get(include=["metadatas"])
        all_pages = set()
        for meta in all_metas.get("metadatas", []):
            if meta and "page" in meta:
                all_pages.add(meta["page"])
        
        sorted_pages = sorted(all_pages, key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else 0)
        print(f"\nPages found in collection: {sorted_pages[:20]}...")
        print(f"Total unique pages: {len(all_pages)}")

print("\n=== PAGE DISTRIBUTION ===")
all_metas = col.get(include=["metadatas"])
page_counts = Counter()

for meta in all_metas.get("metadatas", []):
    if meta and "page" in meta:
        # Normalize to string for counting
        page = str(meta["page"])
        page_counts[page] += 1

print("\nTop 20 pages by document count:")
for page, count in page_counts.most_common(20):
    print(f"  Page {page}: {count} documents")

print("\n=== RECOMMENDATIONS ===")

# Check if page types are mixed
if len(page_types) > 1:
    print("⚠️ ISSUE: Page numbers stored as multiple types (string AND int)")
    print("   Fix: Normalize all page numbers to integers during ingestion")
    print("   The updated code now handles both types during retrieval")
else:
    print("✓ Page numbers are consistently typed")

# Check for page 54
if "54" in page_counts or 54 in [int(p) if p.isdigit() else p for p in page_counts.keys()]:
    print("✓ Page 54 exists in the collection")
else:
    print("⚠️ WARNING: Page 54 not found in collection!")
    print("   This suggests your data ingestion may have skipped this page")
    print("   Check your PDF ingestion logs")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)