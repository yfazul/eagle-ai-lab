from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),  # Now: bolt+ssc://89bd916e.databases.neo4j.io:7687
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

try:
    driver.verify_connectivity()
    print("✓ Neo4j connected successfully!")
    
    with driver.session() as session:
        result = session.run("RETURN 'Hello from Neo4j!' AS message")
        print(f"✓ {result.single()['message']}")
    
    driver.close()
    
except Exception as e:
    print(f"✗ Error: {e}")