import os
import io
import time
import base64
from dotenv import load_dotenv

import fitz                    # PyMuPDF
from PIL import Image
import chromadb
from openai import OpenAI


# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

chroma = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

image_collection = chroma.get_or_create_collection("tn_class6_images")


# ==========================================================
# OCR WITH TIMEOUT + RETRY LOGIC (NO SKIP)
# ==========================================================
def ocr_image(image_bytes: bytes) -> str:

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                timeout=20,  # avoid freezing here
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text exactly as visible in the picture. Do NOT describe the image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            }
                        ]
                    }
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"      ⚠ OCR attempt {attempt+1} failed: {str(e)[:80]}")
            time.sleep(2)

    print("      ✗ OCR failed after 3 attempts.")
    return ""


# ==========================================================
# EMBEDDING
# ==========================================================
def embed_text(text: str):
    emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=384
    )
    return emb.data[0].embedding


# ==========================================================
# MAIN PROCESSOR (NO SKIP)
# ==========================================================
def extract_images_to_chroma(pdf_path, start_page=41, end_page=60):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    print(f"\n📘 PDF Loaded: {pdf_path}")
    print(f"Total pages in PDF: {total_pages}")
    print(f"Processing pages {start_page} to {end_page}...\n")

    # Loop through required pages
    for page_no in range(start_page, min(end_page, total_pages) + 1):

        page_index = page_no - 1
        page = doc.load_page(page_index)
        images = page.get_images(full=True)

        if not images:
            print(f"=== Page {page_no}: No images found ===")
            continue

        print(f"\n=== Page {page_no} | {len(images)} images found ===")

        for img_idx, img_info in enumerate(images):

            img_id = f"p{page_no}_img{img_idx+1}"

            try:
                xref = img_info[0]
                img_dict = doc.extract_image(xref)
                raw_bytes = img_dict["image"]

                # Convert to RGB
                pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                # OCR
                print(f" → OCR {img_id} …")
                ocr_text = ocr_image(image_bytes)

                # Embedding
                embedding = embed_text(ocr_text if ocr_text else "")

                # Store in Chroma
                image_collection.upsert(
                    ids=[img_id],
                    documents=[ocr_text],
                    embeddings=[embedding],
                    metadatas=[{
                        "page": page_no,
                        "image_index": img_idx + 1,
                        "ocr_len": len(ocr_text),
                        "source": pdf_path
                    }]
                )

                print(f" ✓ Stored {img_id} | OCR chars: {len(ocr_text)}")

            except Exception as e:
                print(f"    ✗ Error processing {img_id}: {str(e)[:100]}")
                continue

    doc.close()
    print(f"\n🎉 DONE — Extracted images from pages {start_page} to {end_page} and uploaded to Chroma!")


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    extract_images_to_chroma(PDF_PATH, start_page=52, end_page=52)
