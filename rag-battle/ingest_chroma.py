import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

import chromadb

# -------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY   = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT    = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE  = os.getenv("CHROMA_DATABASE")
NEO4J_URI        = os.getenv("NEO4J_URI")
NEO4J_USERNAME   = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD")

PDF_PATH = "./data/tn_class6_english.pdf"

# -------------------------------------------------------------------
# Clients
# -------------------------------------------------------------------
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# -------------------------------------------------------------------
# Splitter
# -------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    separators=["\n\n", ".", "!", "?", ",", " ", ""]
)

# -------------------------------------------------------------------
# Load PDF
# -------------------------------------------------------------------
print("Loading PDF...")
loader = PyMuPDFLoader(PDF_PATH)
pages = loader.load()

print(f"PDF contains {len(pages)} pages")

# -------------------------------------------------------------------
# Printed page number offset
# -------------------------------------------------------------------
PRINTED_OFFSET = 80  # because PDF 54 → printed 134

# -------------------------------------------------------------------
# Chroma collection
# -------------------------------------------------------------------
vectorstore = Chroma(
    client=client,
    collection_name="tn_class6_text",
    embedding_function=embeddings
)

# -------------------------------------------------------------------
# Ingest
# -------------------------------------------------------------------
total_chunks = 0

for page_doc in pages:
    pdf_page = page_doc.metadata["page"]

    # Convert PDF page → printed textbook page number
    printed_page = pdf_page + PRINTED_OFFSET

    text = page_doc.page_content.strip()
    if not text:
        continue

    # Split into chunks
    chunks = splitter.split_text(text)

    # Convert to LC docs
    lc_docs = [
        Document(
            page_content=c,
            metadata={"pdf_page": pdf_page, "page": printed_page}
        )
        for c in chunks
    ]

    # Chroma IDs
    ids = [f"text_page{printed_page}_{i}" for i in range(len(lc_docs))]
    texts = [d.page_content for d in lc_docs]
    metas = [d.metadata for d in lc_docs]

    # Add to Chroma
    vectorstore.add_texts(
        texts=texts,
        metadatas=metas,
        ids=ids,
    )

    # Add to KG
    for i, d in enumerate(lc_docs):
        graph.query(
            """
            MERGE (t:TextChunk {id: $id})
            SET t.text = $text,
                t.page = $printed_page,
                t.pdf_page = $pdf_page
            """,
            {
                "id": ids[i],
                "text": d.page_content,
                "printed_page": printed_page,
                "pdf_page": pdf_page,
            }
        )

    print(f"PDF page {pdf_page} → Printed page {printed_page}: {len(lc_docs)} chunks")

    total_chunks += len(lc_docs)


print("---------------------------------------------------------")
print(f"Completed! Total chunks: {total_chunks}")
print("Printed-page ingestion completed successfully.")
