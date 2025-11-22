"""
Ingest first 20 pages of TN Class 6 English textbook (PDF)
into:

- Chroma Cloud (text + image OCR embeddings)
- Neo4j Aura (metadata graph)

Pipeline:
1. Extract text from each page, chunk, embed with OpenAI, store in Chroma + Neo4j.
2. Extract images from each page, OCR using DeepSeek, embed with OpenAI, store in Chroma + Neo4j.
"""

import os
import io
from typing import List
import base64

import requests
import fitz                      # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from neo4j import GraphDatabase

# ==========================================================
# 0. LOAD ENV + BASIC CONFIG
# ==========================================================

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "./data/tn_class6_english.pdf")
MAX_PAGES = 20  # only first 20 pages

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Note: Using OpenAI for OCR since DeepSeek doesn't support vision

# --- Chroma Cloud ---
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")  # Fixed: was NEO4J_USER
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Basic sanity checks
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
    raise RuntimeError("Chroma Cloud config missing: CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE")

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise RuntimeError("Neo4j config missing: NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD")


# ==========================================================
# 1. INITIALIZE CLIENTS (OpenAI, Chroma Cloud, Neo4j)
# ==========================================================

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma Cloud client
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# Create / get collections in Chroma Cloud
# Delete existing collections if they exist with wrong dimensions
try:
    chroma_client.delete_collection("tn_class6_text")
    chroma_client.delete_collection("tn_class6_images")
    print("✓ Deleted existing collections")
except:
    pass

# Create fresh collections (will use OpenAI's 1536 dimensions)
text_collection = chroma_client.get_or_create_collection("tn_class6_text")
image_collection = chroma_client.get_or_create_collection("tn_class6_images")

print("✓ Collections created/verified")

# Neo4j driver - Fixed: Using correct URI format
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,  # Should be: bolt+ssc://xxxxx.databases.neo4j.io:7687
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

print("✓ All clients initialized successfully")


# ==========================================================
# 2. HELPER: OpenAI Embedding
# ==========================================================

def embed_openai(text: str) -> List[float]:
    """
    Create embedding with OpenAI text-embedding-3-small.
    Returns a list[float] ready for Chroma.
    Note: text-embedding-3-small = 1536 dimensions
          text-embedding-ada-002 = 1536 dimensions
          Can specify dimensions parameter to reduce size
    """
    text = text.strip()
    if not text:
        return []

    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=384  # Reduce to 384 dimensions to match existing collection
    )
    return resp.data[0].embedding


# ==========================================================
# 3. HELPER: Text chunking for RAG
# ==========================================================

def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> List[str]:
    """
    Simple word-based overlapping chunker:
    - max_words: approximate length of each chunk
    - overlap: number of words overlapped between adjacent chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += max_words - overlap

    return chunks


# ==========================================================
# 4. HELPER: OCR using OpenAI GPT-4 Vision
# ==========================================================

def ocr_image(image_bytes: bytes) -> str:
    """
    Extract text from images using OpenAI GPT-4 Vision.
    Most reliable OCR solution.
    """
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the text content, no descriptions or explanations. If there's no text, return an empty response."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0
        )
        
        text = response.choices[0].message.content
        return text.strip() if text else ""
        
    except Exception as e:
        print(f"    • OCR error: {str(e)[:100]}")
        return ""


# ==========================================================
# 5. NEO4J HELPERS
# ==========================================================

def init_neo4j_constraints() -> None:
    """
    Create basic uniqueness constraints for TextChunk and ImageChunk.
    This makes MERGE operations efficient and avoids duplicates.
    """
    try:
        with neo4j_driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:ImageChunk) REQUIRE i.id IS UNIQUE"
            )
        print("✓ Neo4j constraints initialized")
    except Exception as e:
        print(f"✗ Error creating Neo4j constraints: {e}")
        raise


def create_text_node(session, chunk_id: str, page: int, text: str) -> None:
    """
    Create or update a TextChunk node in Neo4j representing a text chunk.
    """
    session.run(
        """
        MERGE (t:TextChunk {id: $id})
        SET t.page = $page,
            t.text = $text,
            t.word_count = size(split($text, ' '))
        """,
        id=chunk_id,
        page=page,
        text=text,
    )


def create_image_node(session, img_id: str, page: int, ocr_text: str, image_path: str) -> None:
    """
    Create or update an ImageChunk node in Neo4j representing an image/diagram.
    """
    session.run(
        """
        MERGE (i:ImageChunk {id: $id})
        SET i.page = $page,
            i.ocr_text = $ocr_text,
            i.image_path = $image_path,
            i.has_text = CASE WHEN $ocr_text <> '' THEN true ELSE false END
        """,
        id=img_id,
        page=page,
        ocr_text=ocr_text,
        image_path=image_path,
    )


# ==========================================================
# 6. MAIN INGESTION FUNCTION
# ==========================================================

def ingest_pdf(pdf_path: str, max_pages: int = 20) -> None:
    """
    Ingest the first `max_pages` from the TN Class 6 English PDF:
      - For each page:
        * Extracts text
        * Chunks text
        * Embeds text chunks with OpenAI
        * Stores in Chroma Cloud + Neo4j
        * Extracts images
        * OCR via DeepSeek
        * Embeds OCR text with OpenAI
        * Stores in Chroma Cloud + Neo4j
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"\nOpening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = min(max_pages, doc.page_count)
    print(f"Total pages in PDF: {doc.page_count}, processing first {total_pages} pages\n")

    # Initialize Neo4j constraints
    init_neo4j_constraints()

    # Verify Neo4j connection
    try:
        neo4j_driver.verify_connectivity()
        print("✓ Neo4j connection verified\n")
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        raise

    with neo4j_driver.session() as session:
        for page_index in range(total_pages):
            page_num = page_index + 1
            page = doc.load_page(page_index)
            print(f"=== Page {page_num} ===")

            # ---------------- TEXT INGESTION ----------------
            raw_text = page.get_text("text")
            # Basic whitespace cleanup
            raw_text = " ".join(raw_text.split())

            if raw_text.strip():
                chunks = chunk_text(raw_text, max_words=200, overlap=40)
                print(f"  • Found {len(chunks)} text chunks")

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"p{page_num}_t{idx+1}"

                    try:
                        # Create OpenAI embedding
                        embedding = embed_openai(chunk)

                        # Store in Chroma Cloud collection for text
                        text_collection.add(
                            ids=[chunk_id],
                            documents=[chunk],
                            metadatas=[{"page": page_num, "chunk_index": idx + 1}],
                            embeddings=[embedding],
                        )

                        # Store as a node in Neo4j
                        create_text_node(session, chunk_id, page_num, chunk)
                        
                    except Exception as e:
                        print(f"    ✗ Error processing text chunk {chunk_id}: {e}")
                        continue
            else:
                print("  • No text detected on this page.")

            # ---------------- IMAGE INGESTION ----------------
            images = page.get_images(full=True)
            print(f"  • Found {len(images)} images")

            for img_index, img_info in enumerate(images):
                xref = img_info[0]
                img_id = f"p{page_num}_img{img_index+1}"

                try:
                    # Extract image with better error handling for colorspaces
                    base_pix = fitz.Pixmap(doc, xref)
                    
                    # Strategy: Always convert to RGB through PIL for maximum compatibility
                    try:
                        # Method 1: Try to get RGB directly
                        if base_pix.colorspace:
                            if base_pix.colorspace.name in ["DeviceRGB", "DeviceGray"]:
                                # Already in a good format
                                pix = base_pix
                            else:
                                # Convert to RGB
                                pix = fitz.Pixmap(fitz.csRGB, base_pix)
                                base_pix = None
                        else:
                            # No colorspace info, try to convert anyway
                            pix = fitz.Pixmap(fitz.csRGB, base_pix)
                            base_pix = None
                        
                        # Get image data
                        img_data = pix.samples
                        img_mode = "RGB" if pix.n >= 3 else "L"  # RGB or Grayscale
                        
                        # Create PIL Image
                        pil_img = Image.frombytes(img_mode, [pix.width, pix.height], img_data)
                        
                        # Ensure RGB mode for consistency
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        print(f"    • Colorspace conversion failed, trying alternative method: {str(e)[:50]}")
                        
                        # Method 2: Extract as bytes and let PIL handle it
                        try:
                            # Get the raw image data from PDF
                            img_dict = doc.extract_image(xref)
                            img_data = img_dict["image"]
                            
                            # Load with PIL which handles more formats
                            pil_img = Image.open(io.BytesIO(img_data))
                            
                            # Convert to RGB
                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                                
                        except Exception as e2:
                            print(f"    • Alternative method also failed: {str(e2)[:50]}")
                            # Skip this image
                            continue
                    
                    # Convert PIL image to PNG bytes
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()

                    # Save image to disk (optional but useful for UI debugging)
                    out_dir = "./data/extracted_images"
                    os.makedirs(out_dir, exist_ok=True)
                    img_path = os.path.join(out_dir, f"{img_id}.png")
                    pil_img.save(img_path)

                    # Run OCR with OpenAI Vision
                    print(f"    • Running OCR on {img_id}...")
                    ocr_text = ocr_image(image_bytes)
                    
                    if ocr_text:
                        print(f"    • OCR extracted {len(ocr_text)} characters")
                    else:
                        print(f"    • No text found in image")

                    # Create embedding from OCR text (if any text)
                    embedding = embed_openai(ocr_text) if ocr_text else None

                    # Store in Chroma Cloud collection for images
                    image_collection.add(
                        ids=[img_id],
                        documents=[ocr_text if ocr_text else ""],
                        metadatas=[{
                            "page": page_num,
                            "image_index": img_index + 1,
                            "image_path": img_path,
                            "ocr_len": len(ocr_text) if ocr_text else 0,
                        }],
                        embeddings=[embedding] if embedding else None,
                    )

                    # Store as a node in Neo4j
                    create_image_node(session, img_id, page_num, ocr_text or "", img_path)

                except Exception as e:
                    print(f"    ✗ Error processing image {img_id}: {e}")
                    continue

            print()  # Blank line between pages

    doc.close()
    print("\n✅ Ingestion completed successfully (text + images into Chroma Cloud + Neo4j).")


# ==========================================================
# 7. ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    try:
        ingest_pdf(PDF_PATH, max_pages=MAX_PAGES)
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up connections
        try:
            neo4j_driver.close()
            print("\n✓ Neo4j connection closed")
        except:
            pass