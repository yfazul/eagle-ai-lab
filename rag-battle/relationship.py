import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ==========================================================
# 1. CREATE NEXT RELATIONSHIPS BETWEEN TEXT CHUNKS
# ==========================================================
def create_next_relationships():
    cypher = """
    MATCH (t:TextChunk)
    WITH t.page AS page, t
    ORDER BY page, t.chunk_index
    WITH collect(t) AS chunks
    UNWIND range(0, size(chunks)-2) AS i
    WITH chunks[i] AS a, chunks[i+1] AS b
    MERGE (a)-[:NEXT]->(b)
    """

    with driver.session() as session:
        session.run(cypher)
        print("✓ NEXT relationships created")


# ==========================================================
# 2. LINK TEXT & IMAGES WITH SAME_PAGE
# ==========================================================
def create_same_page_relationships():
    cypher = """
    MATCH (t:TextChunk), (i:ImageChunk)
    WHERE t.page = i.page
    MERGE (t)-[:SAME_PAGE]->(i)
    """

    with driver.session() as session:
        session.run(cypher)
        print("✓ SAME_PAGE relationships created")


# ==========================================================
# 3. USE LLM TO EXTRACT ENTITIES AND CREATE MENTIONS
# ==========================================================
def extract_entities(text):
    prompt = f"""
    Extract named entities (characters, places, objects)
    from this text. Output as a comma-separated list, no extra words:

    "{text}"
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    result = resp.choices[0].message.content.strip()
    return [x.strip() for x in result.split(",") if x.strip()]


def create_mention_relationships():
    with driver.session() as session:
        text_nodes = session.run("MATCH (t:TextChunk) RETURN t.id AS id, t.text AS text")

        for record in text_nodes:
            tid = record["id"]
            text = record["text"]

            entities = extract_entities(text)

            for ent in entities:
                session.run("""
                MERGE (e:Entity {name:$name})
                WITH e
                MATCH (t:TextChunk {id:$tid})
                MERGE (t)-[:MENTIONS]->(e)
                """, name=ent, tid=tid)

            print(f"✓ MENTIONS for {tid}: {entities}")


# ==========================================================
# EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("\n=== Generating Knowledge Graph Relationships ===\n")

    create_next_relationships()
    create_same_page_relationships()
    create_mention_relationships()

    print("\n🎉 KG enrichment complete!\n")
