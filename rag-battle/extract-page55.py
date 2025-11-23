import fitz  # PyMuPDF
import os

PDF_PATH = "./data/tn_class6_english.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

doc = fitz.open(PDF_PATH)

page_number = 55
page_index = page_number - 1   # convert to 0-based index

page = doc.load_page(page_index)

print("==============================================")
print(f"🔍 TEXT FROM PAGE {page_number}")
print("==============================================")
print(page.get_text("text"))
