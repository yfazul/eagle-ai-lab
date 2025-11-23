"""
Quick Neo4j Reset - Drop ALL indexes, constraints, and optionally data
Run this before starting ingestion if you're having conflicts
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def run_query(query, description=""):
    """Execute a query"""
    try:
        with driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]
            if description:
                print(f"✓ {description}")
            return data
    except Exception as e:
        print(f"✗ {description}: {e}")
        return None

print("="*60)
print("QUICK NEO4J RESET")
print("="*60)

# 1. Drop all constraints
print("\nDropping all constraints...")
constraints = run_query("SHOW CONSTRAINTS")
if constraints:
    for cons in constraints:
        name = cons.get('name')
        if name:
            run_query(f"DROP CONSTRAINT `{name}` IF EXISTS", f"Dropped: {name}")
else:
    print("No constraints found")

# 2. Drop all indexes
print("\nDropping all indexes...")
indexes = run_query("SHOW INDEXES")
if indexes:
    for idx in indexes:
        name = idx.get('name')
        if name:
            run_query(f"DROP INDEX `{name}` IF EXISTS", f"Dropped: {idx['name']}")
else:
    print("No indexes found")

# 3. Optional: Clear all data
print("\n" + "="*60)
clear_data = input("Do you want to DELETE ALL DATA too? (yes/no): ").strip().lower()

if clear_data == 'yes':
    result = run_query("MATCH (n) RETURN count(n) as count")
    if result and result[0]['count'] > 0:
        print(f"\nDeleting {result[0]['count']} nodes...")
        
        # Delete in batches
        deleted_total = 0
        while True:
            result = run_query(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) as deleted"
            )
            if result and result[0]['deleted'] > 0:
                deleted_total += result[0]['deleted']
                print(f"  Deleted {deleted_total} nodes...")
            else:
                break
        
        print(f"✓ All {deleted_total} nodes deleted")
    else:
        print("No data to delete")

driver.close()

print("\n" + "="*60)
print("✓ RESET COMPLETE")
print("="*60)
print("\nYou can now run: python file_ingestion.py")