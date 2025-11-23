import fitz
import base64
import os
from PIL import Image
import io
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Load env vars
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

PDF_PATH = "./data/tn_class6_english.pdf"
PAGE_NUMBER = 60  # test only page 60

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma Client
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# Ensure collection exists
collection = chroma_client.get_or_create_collection("tn_class6_images")


# OCR helper
def ocr_image(image_bytes: bytes) -> str:
    base64_img = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


# MAIN
def ocr_page60():
    doc = fitz.open(PDF_PATH)
    page = doc.load_page(PAGE_NUMBER - 1)

    images = page.get_images(full=True)
    print(f"Found {len(images)} images on page 60.")

    for idx, img in enumerate(images):
        xref = img[0]

        # Extract raw image
        base_pix = fitz.Pixmap(doc, xref)

        # Convert to PNG using PIL
        if base_pix.n >= 3:
            img_mode = "RGB"
        else:
            img_mode = "L"

        pil_img = Image.frombytes(img_mode, [base_pix.width, base_pix.height], base_pix.samples)

        # Save as PNG bytes
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        image_bytes = img_buf.getvalue()

        print(f"OCR image {idx+1}...")

        # OCR
        ocr_text = ocr_image(image_bytes)
        print("OCR RESULT:")
        print(ocr_text)
        print("----------------")

        # Store in Chroma
        collection.add(
            ids=[f"p60_img{idx+1}"],
            documents=[ocr_text],
            metadatas=[{"page": 60, "image_index": idx + 1}],
        )

    doc.close()
    print("✓ Page 60 OCR stored in Chroma.")


if __name__ == "__main__":
    ocr_page60()
