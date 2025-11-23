import fitz
import os

PDF_PATH = "data/tn_class6_english.pdf"   # your file
OUTPUT_DIR = "data/page64_images"
PRINTED_PAGE = 64                    # printed page number
PDF_OFFSET = 0                       # adjust if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)
pdf_page = PRINTED_PAGE - 1 + PDF_OFFSET

print(f"Extracting images from printed page {PRINTED_PAGE} (PDF page {pdf_page+1})...")

page = doc[pdf_page]
image_list = page.get_images(full=True)

if not image_list:
    print("❌ No images found on this page.")
    exit()

print(f"🖼 Found {len(image_list)} images.")

for idx, img in enumerate(image_list):
    xref = img[0]
    pix = fitz.Pixmap(doc, xref)

    out_path = os.path.join(OUTPUT_DIR, f"page64_img_{idx}.png")

    # Handle CMYK or mask images
    if pix.n in (1, 3, 4):
        # 1 = grayscale, 3 = RGB, 4 = CMYK (unsupported)
        if pix.n == 4:  # CMYK
            pix = fitz.Pixmap(fitz.csRGB, pix)
    else:
        # Indexed, or weird mode -> convert to RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)

    pix.save(out_path)
    pix = None

    print(f"   ✔ Saved {out_path}")

print("🎉 Done! All images extracted.")
