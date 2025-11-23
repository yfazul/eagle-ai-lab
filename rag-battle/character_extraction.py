import os
import re
from dotenv import load_dotenv
import chromadb
from langchain_core.documents import Document
from neo4j import GraphDatabase
from openai import OpenAI

# =============================
# LOAD ENV VARS
# =============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# CONNECT TO CHROMA
# =============================
chroma = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

collection = chroma.get_or_create_collection("tn_class6_text")

# =============================
# CONNECT TO NEO4J
# =============================
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# =============================
# SIMPLE REGEX PERSON GUESSER
# =============================
def extract_character_candidates(text: str):
    """
    Basic regex: capitalized words (Mani, Amma, Selvi, Raju...)
    This works well for your textbook.
    """
    pattern = r"\b([A-Z][a-z]{2,20})\b"
    names = re.findall(pattern, text)
    return list(set(names))


# =============================
# OPTIONAL GPT-BASED CHARACTER NER
# =============================
def llm_extract_characters(text: str):
    """
    Use GPT-4o-mini to extract ONLY real characters (names).
    """
    prompt = f"""
Extract ONLY the characters (names of people) mentioned in this text.
Return a comma-separated list. No other words.

TEXT:
{text}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = resp.choices[0].message["content"]
        names = [n.strip() for n in raw.split(",") if n.strip()]
        return names
    except Exception:
        return []


# =============================
# UPSERT CHARACTER NODE
# =============================
def upsert_character(tx, name):
    tx.run(
        """
        MERGE (c:Character {name: $name})
        """,
        name=name
    )


# =============================
# CREATE MENTIONS EDGE
# =============================
def create_mentions_edge(tx, chunk_id, char_name):
    tx.run(
        """
        MATCH (t:TextChunk {id: $chunk_id})
        MATCH (c:Character {name: $char_name})
        MERGE (t)-[:MENTIONS]->(c)
        """,
        chunk_id=chunk_id,
        char_name=char_name
    )


# =============================
# MAIN EXTRACTOR
# =============================
def extract_and_store_characters(use_llm=False):
    print("🔍 Fetching all Chroma text documents...")

    res = collection.get(include=["documents", "metadatas"])
    docs = res["documents"]
    metas = res["metadatas"]
    ids = res["ids"]

    print(f"📄 Retrieved {len(docs)} text chunks from Chroma.")

    all_characters = set()

    with driver.session() as session:
        for chunk_text, meta, chunk_id in zip(docs, metas, ids):

            if not chunk_text:
                continue

            # Extract characters
            if use_llm:
                candidates = llm_extract_characters(chunk_text)
            else:
                candidates = extract_character_candidates(chunk_text)

            candidates = [c for c in candidates if len(c) > 1]

            if not candidates:
                continue

            print(f"➡️ Chunk {chunk_id} -> Found: {candidates}")

            for name in candidates:
                all_characters.add(name)

                # Insert Character node
                session.write_transaction(upsert_character, name)

                # Create MENTIONS edge
                session.write_transaction(create_mentions_edge, chunk_id, name)

    print("\n============================")
    print("🎉 Character Extraction Complete!")
    print("============================")
    print("Characters found:")
    print(all_characters)
    print(f"Total unique characters: {len(all_characters)}")


# =============================
# RUN
# =============================
if __name__ == "__main__":
    # Set use_llm=True for cleaner character detection
    extract_and_store_characters(use_llm=False)
